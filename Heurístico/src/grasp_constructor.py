"""
Fase Constructiva GRASP (Soporta Reparación VNS)
"""
import random
import time
import pandas as pd
from typing import List, Dict, Tuple, Any
from src.data_structures import SolucionGlobal, PATRONES_VISITA

def evaluar_mejor_insercion_dia(solucion: SolucionGlobal, nodo_id: str, demanda_kg: float, dia: str) -> Tuple[bool, int, int, float, float]:
    """Busca la mejor ruta y posición minimizando el incremento de distancia puro."""
    mejor_delta_dist = float('inf')
    mejor_delta_time = 0.0
    mejor_vehiculo_idx = -1
    mejor_pos = -1
    es_factible = False

    for v_idx, ruta in enumerate(solucion.rutas[dia]):
        for pos in range(1, len(ruta.nodos)):
            factible, d_dist, d_time = ruta.simular_insercion(nodo_id, demanda_kg, pos)
            
            if factible and d_dist < mejor_delta_dist:
                mejor_delta_dist = d_dist
                mejor_delta_time = d_time
                mejor_vehiculo_idx = v_idx
                mejor_pos = pos
                es_factible = True

    return es_factible, mejor_vehiculo_idx, mejor_pos, mejor_delta_dist, mejor_delta_time

def reparar_solucion_grasp(solucion: SolucionGlobal, df_demanda: pd.DataFrame, alpha: float) -> bool:
    """
    Toma una solución (parcial o vacía) y asigna los nodos_no_asignados usando GRASP con memoria.
    Devuelve True si logró asignar todos los nodos, False si quedó alguno huérfano (infactible).
    """
    frecuencia_nodos = {row['Cod. Contenedor']: int(row['Opt_Frecuencia_Semanal']) for _, row in df_demanda.iterrows()}
    
    # 1. Inicialización de la Caché de Memoria para los nodos huérfanos
    memoria_evaluaciones = {nodo: {} for nodo in solucion.nodos_no_asignados}
    for nodo in solucion.nodos_no_asignados:
        for dia in solucion.dias:
            memoria_evaluaciones[nodo][dia] = evaluar_mejor_insercion_dia(solucion, nodo, solucion.demandas_diarias[nodo], dia)
            
    # 2. Bucle de Inserción
    while solucion.nodos_no_asignados:
        candidatos = []
        
        for nodo_id in solucion.nodos_no_asignados:
            frecuencia = frecuencia_nodos[nodo_id]
            for patron in PATRONES_VISITA[frecuencia]:
                factible_patron = True
                delta_dist_total = 0.0
                inserciones_dias = {}

                for dia in patron:
                    f_dia, v_idx, pos, d_dist, d_time = memoria_evaluaciones[nodo_id][dia]
                    if not f_dia:
                        factible_patron = False
                        break 
                    
                    delta_dist_total += d_dist
                    inserciones_dias[dia] = {'v_idx': v_idx, 'pos': pos, 'd_dist': d_dist, 'd_time': d_time}

                if factible_patron:
                    candidatos.append({
                        'nodo_id': nodo_id,
                        'patron': patron,
                        'demanda_kg': solucion.demandas_diarias[nodo_id],
                        'delta_dist_total': delta_dist_total,
                        'inserciones_dias': inserciones_dias
                    })

        if not candidatos:
            break # Imposible insertar más nodos

        # RCL y Selección
        c_min = min(c['delta_dist_total'] for c in candidatos)
        c_max = max(c['delta_dist_total'] for c in candidatos)
        umbral_rcl = c_min + alpha * (c_max - c_min)

        rcl = [c for c in candidatos if c['delta_dist_total'] <= umbral_rcl]
        elegido = random.choice(rcl)

        # Aplicar inserción
        nodo_id = elegido['nodo_id']
        dias_modificados = []
        
        for dia, info in elegido['inserciones_dias'].items():
            ruta_objetivo = solucion.rutas[dia][info['v_idx']]
            ruta_objetivo.aplicar_insercion(nodo_id, elegido['demanda_kg'], info['pos'], info['d_dist'], info['d_time'])
            dias_modificados.append(dia)

        solucion.nodos_no_asignados.remove(nodo_id)
        del memoria_evaluaciones[nodo_id] 
        
        # Actualización de la memoria
        for nodo in solucion.nodos_no_asignados:
            for dia in dias_modificados:
                memoria_evaluaciones[nodo][dia] = evaluar_mejor_insercion_dia(solucion, nodo, solucion.demandas_diarias[nodo], dia)

    solucion.calcular_coste_objetivo()
    return len(solucion.nodos_no_asignados) == 0


def construir_solucion_grasp(df_dist: pd.DataFrame, df_time: pd.DataFrame, df_demanda: pd.DataFrame, num_vehiculos: int, alpha: float, semilla: int) -> SolucionGlobal:
    """Orquestador original. Crea una solución vacía y llama al motor de reparación."""
    random.seed(semilla)
    
    demandas_diarias = {}
    for _, row in df_demanda.iterrows():
        carga = min(row['Tasa_Pesimista_Kg_Dia'] * (7.0 / row['Opt_Frecuencia_Semanal']), row['Umbral_Recogida_Efectivo'])
        demandas_diarias[row['Cod. Contenedor']] = carga

    solucion = SolucionGlobal(df_dist, df_time, demandas_diarias, num_vehiculos)
    reparar_solucion_grasp(solucion, df_demanda, alpha)
    return solucion