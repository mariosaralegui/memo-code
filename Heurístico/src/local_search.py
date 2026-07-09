"""
Operadores de Búsqueda Local
"""
from src.data_structures import Ruta

def apply_2opt(ruta: Ruta) -> bool:
    """
    Operador 2-opt (Adaptado para Grafos Asimétricos).
    Invierte el orden de un segmento de la ruta.
    """
    if len(ruta.nodos) <= 4:
        return False
        
    mejora_global = False
    mejora_iteracion = True
    n = len(ruta.nodos)
    
    while mejora_iteracion:
        mejora_iteracion = False
        
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Nodos de conexión
                prev_i = ruta.nodos[i - 1]
                nodo_i = ruta.nodos[i]
                nodo_j = ruta.nodos[j]
                next_j = ruta.nodos[j + 1]
                
                # 1. Coste de los bordes eliminados
                coste_bordes_eliminados = ruta.df_dist.loc[prev_i, nodo_i] + ruta.df_dist.loc[nodo_j, next_j]
                
                # 2. Coste del segmento interno actual (A -> B -> C)
                segmento_actual = ruta.nodos[i:j+1]
                coste_interno_actual = sum(ruta.df_dist.loc[segmento_actual[k], segmento_actual[k+1]] for k in range(len(segmento_actual)-1))
                
                # 3. Coste de los nuevos bordes (cruzados)
                coste_bordes_nuevos = ruta.df_dist.loc[prev_i, nodo_j] + ruta.df_dist.loc[nodo_i, next_j]
                
                # 4. Coste del segmento interno invertido (C -> B -> A)
                segmento_invertido = segmento_actual[::-1]
                coste_interno_invertido = sum(ruta.df_dist.loc[segmento_invertido[k], segmento_invertido[k+1]] for k in range(len(segmento_invertido)-1))
                
                delta_dist = (coste_bordes_nuevos + coste_interno_invertido) - (coste_bordes_eliminados + coste_interno_actual)
                
                if delta_dist < -0.001:
                    # Aplicar inversión
                    ruta.nodos[i:j+1] = segmento_invertido
                    mejora_iteracion = True
                    mejora_global = True
                    break
            if mejora_iteracion:
                break
                
    if mejora_global:
        ruta.recalcular_metricas()
    return mejora_global


def apply_or_opt(ruta: Ruta, k_size: int) -> bool:
    """
    Operador Generalizado Or-opt. Extrae una cadena de tamaño k_size y la reubica.
    k_size = 1 (Relocate clásico)
    k_size = 2 (Mueve 2 nodos juntos)
    k_size = 3 (Mueve 3 nodos juntos)
    """
    if len(ruta.nodos) <= k_size + 2:
        return False
        
    mejora_global = False
    mejora_iteracion = True
    
    while mejora_iteracion:
        mejora_iteracion = False
        n = len(ruta.nodos)
        
        for i in range(1, n - k_size):
            # 1. Identificar el bloque y sus bordes originales
            prev_i = ruta.nodos[i - 1]
            first_k = ruta.nodos[i]
            last_k = ruta.nodos[i + k_size - 1]
            next_k = ruta.nodos[i + k_size]
            
            # 2. Calcular impacto de EXTIRPAR el bloque
            dist_extraida = ruta.df_dist.loc[prev_i, first_k] + ruta.df_dist.loc[last_k, next_k]
            dist_cierre = ruta.df_dist.loc[prev_i, next_k]
            
            # 3. Simular la ruta SIN el bloque
            ruta_sin_bloque = ruta.nodos[:i] + ruta.nodos[i + k_size:]
            
            for idx_insert in range(1, len(ruta_sin_bloque)):
                # No tiene sentido reinsertar el bloque exactamente donde estaba
                if idx_insert == i:
                    continue
                    
                # 4. Identificar los nuevos vecinos en la ruta simulada
                new_prev = ruta_sin_bloque[idx_insert - 1]
                new_next = ruta_sin_bloque[idx_insert]
                
                # 5. Calcular impacto de INSERTAR el bloque
                dist_rota = ruta.df_dist.loc[new_prev, new_next]
                dist_creada = ruta.df_dist.loc[new_prev, first_k] + ruta.df_dist.loc[last_k, new_next]
                
                # Balance de kilómetros: (+) lo que añadimos, (-) lo que quitamos
                delta_dist = (dist_cierre + dist_creada) - (dist_extraida + dist_rota)
                
                if delta_dist < -0.001:
                    # Aplicar la mutación matemáticamente segura
                    bloque = ruta.nodos[i : i + k_size]
                    ruta.nodos = ruta_sin_bloque[:idx_insert] + bloque + ruta_sin_bloque[idx_insert:]
                    
                    mejora_iteracion = True
                    mejora_global = True
                    break # Rompe el bucle de inserción
            if mejora_iteracion:
                break # Rompe el bucle de extracción para volver a empezar desde cero
                
    if mejora_global:
        ruta.recalcular_metricas()
    return mejora_global


def apply_swap(ruta: Ruta) -> bool:
    """Intercambia dos nodos de posición."""
    if len(ruta.nodos) <= 4:
        return False
        
    mejora_global = False
    mejora_iteracion = True
    
    while mejora_iteracion:
        mejora_iteracion = False
        n = len(ruta.nodos)
        
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                nodo_i, nodo_j = ruta.nodos[i], ruta.nodos[j]
                
                if i + 1 == j: # Adyacentes
                    delta_dist = (
                        ruta.df_dist.loc[ruta.nodos[i-1], nodo_j] + ruta.df_dist.loc[nodo_j, nodo_i] + ruta.df_dist.loc[nodo_i, ruta.nodos[j+1]] -
                        (ruta.df_dist.loc[ruta.nodos[i-1], nodo_i] + ruta.df_dist.loc[nodo_i, nodo_j] + ruta.df_dist.loc[nodo_j, ruta.nodos[j+1]])
                    )
                else: # No adyacentes
                    delta_dist = (
                        ruta.df_dist.loc[ruta.nodos[i-1], nodo_j] + ruta.df_dist.loc[nodo_j, ruta.nodos[i+1]] +
                        ruta.df_dist.loc[ruta.nodos[j-1], nodo_i] + ruta.df_dist.loc[nodo_i, ruta.nodos[j+1]] -
                        (ruta.df_dist.loc[ruta.nodos[i-1], nodo_i] + ruta.df_dist.loc[nodo_i, ruta.nodos[i+1]] +
                         ruta.df_dist.loc[ruta.nodos[j-1], nodo_j] + ruta.df_dist.loc[nodo_j, ruta.nodos[j+1]])
                    )
                    
                if delta_dist < -0.001:
                    ruta.nodos[i], ruta.nodos[j] = ruta.nodos[j], ruta.nodos[i]
                    mejora_iteracion = True
                    mejora_global = True
                    break
            if mejora_iteracion:
                break
                
    if mejora_global:
        ruta.recalcular_metricas()
    return mejora_global

# Wrappers para el VND
def op_2opt(ruta): return apply_2opt(ruta)
def op_or_opt_1(ruta): return apply_or_opt(ruta, 1)
def op_or_opt_2(ruta): return apply_or_opt(ruta, 2)
def op_or_opt_3(ruta): return apply_or_opt(ruta, 3)
def op_swap(ruta): return apply_swap(ruta)