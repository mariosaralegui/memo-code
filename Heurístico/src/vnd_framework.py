"""
Variable Neighborhood Descent (VND Configurable)
"""
from src.data_structures import SolucionGlobal
from src.local_search import apply_2opt, apply_or_opt, apply_swap

def get_operadores_dinamicos(use_2opt: bool, max_or_opt_k: int, use_swap: bool):
    """
    Construye la lista de vecindarios dinámicamente.
    Orden de ejecución: [2-opt] -> [Or-opt(1)...Or-opt(k)] -> [Swap]
    """
    operadores = []
    
    if use_2opt:
        operadores.append(apply_2opt)
        
    if max_or_opt_k > 0:
        # Usamos current_k=k en la lambda para fijar el valor exacto en el bucle
        for k in range(1, max_or_opt_k + 1):
            operadores.append(lambda r, current_k=k: apply_or_opt(r, current_k))
            
    if use_swap:
        operadores.append(apply_swap)
        
    return operadores

def optimize_route_vnd(ruta, operadores) -> bool:
    """Aplica la secuencia determinista de operadores hasta llegar a un óptimo local."""
    k = 0
    k_max = len(operadores)
    mejora_global = False
    
    while k < k_max:
        hubo_mejora = operadores[k](ruta)
        if hubo_mejora:
            mejora_global = True
            k = 0 # Reiniciar la búsqueda al vecindario más pequeño (Intensificación)
        else:
            k += 1 # Ampliar el vecindario (Diversificación)
            
    return mejora_global

def run_vnd_global(solucion: SolucionGlobal, use_2opt: bool = True, max_or_opt_k: int = 3, use_swap: bool = False) -> SolucionGlobal:
    """Itera sobre la semana aplicando la configuración exacta del VND elegida."""
    operadores = get_operadores_dinamicos(use_2opt, max_or_opt_k, use_swap)
    
    if not operadores:
        return solucion # Sin operadores, devuelve la ruta intacta
        
    for dia in solucion.dias:
        for ruta in solucion.rutas[dia]:
            if len(ruta.nodos) > 3: 
                optimize_route_vnd(ruta, operadores)
                    
    solucion.calcular_coste_objetivo()
    return solucion