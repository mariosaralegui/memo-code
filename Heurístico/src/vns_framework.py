"""
Variable Neighborhood Search (VNS-GRASP)
"""
import time
import copy
import random
import pandas as pd
from src.data_structures import SolucionGlobal
from src.grasp_constructor import reparar_solucion_grasp
from src.vnd_framework import run_vnd_global

def shake_solution(solucion: SolucionGlobal, k_fraction: float) -> SolucionGlobal:
    """Fase de Destrucción (Shake)."""
    nodos_asignados = list(set(solucion.demandas_diarias.keys()) - solucion.nodos_no_asignados)
    if not nodos_asignados:
        return solucion
        
    num_a_destruir = max(1, int(len(nodos_asignados) * k_fraction))
    nodos_a_borrar = set(random.sample(nodos_asignados, num_a_destruir))
    
    for dia in solucion.dias:
        for ruta in solucion.rutas[dia]:
            nodos_limpios = ['PIRS']
            nueva_carga_real = 0.0 
            
            for nodo in ruta.nodos[1:-1]:
                if nodo not in nodos_a_borrar:
                    nodos_limpios.append(nodo)
                    nueva_carga_real += solucion.demandas_diarias[nodo] 
                    
            nodos_limpios.append('PIRS')
            
            if len(nodos_limpios) != len(ruta.nodos):
                ruta.nodos = nodos_limpios
                ruta.carga_actual = nueva_carga_real
                ruta.recalcular_metricas()
                
    solucion.nodos_no_asignados.update(nodos_a_borrar)
    solucion.calcular_coste_objetivo()
    return solucion

def run_vns_global(solucion_inicial: SolucionGlobal, df_demanda: pd.DataFrame, time_limit_sec: int, vnd_params: dict, max_iter_sin_mejora: int = 50, logger=None):
    """
    Bucle principal VNS. 
    RETORNA: (Mejor_Solucion, Historial_Convergencia)
    """
    vecindarios_k = [0.05, 0.10, 0.15, 0.20, 0.25]
    k_max = len(vecindarios_k)
    
    x = copy.deepcopy(solucion_inicial)
    coste_x = x.calcular_coste_objetivo()
    
    x_mejor = copy.deepcopy(x)
    coste_mejor = coste_x
    
    iter_sin_mejora = 0
    start_time = time.time()
    iteracion_total = 0
    


    historial_convergencia = []
    # Guardamos el punto de partida (T=0)
    historial_convergencia.append({
        'Iteracion': 0, 'Tiempo_s': 0.0, 'FO_Mejor': coste_mejor, 'Shake_%': 0.0
    })
    
    if logger: logger.info(f"   [VNS] Iniciando ciclo (Límite: {time_limit_sec}s o {max_iter_sin_mejora} iter sin mejora)...")

    while iter_sin_mejora < max_iter_sin_mejora and (time.time() - start_time) < time_limit_sec:
        k = 0
        while k < k_max:
            # Control estricto de tiempo dentro del bucle de vecindarios
            if (time.time() - start_time) >= time_limit_sec:
                break
                
            iteracion_total += 1
            x_prima = copy.deepcopy(x)
            x_prima = shake_solution(x_prima, vecindarios_k[k])
            exito_reparacion = reparar_solucion_grasp(x_prima, df_demanda, alpha=0.1)
            
            if exito_reparacion:
                x_dos_prima = run_vnd_global(
                    x_prima, 
                    use_2opt=vnd_params['use_2opt'], 
                    max_or_opt_k=vnd_params['max_or_opt_k'], 
                    use_swap=vnd_params['use_swap']
                )
                coste_x_dos_prima = x_dos_prima.coste_total_km
                
                if coste_x_dos_prima < coste_x - 0.001:
                    x = copy.deepcopy(x_dos_prima)
                    coste_x = coste_x_dos_prima
                    
                    if coste_x < coste_mejor - 0.001:
                        x_mejor = copy.deepcopy(x)
                        coste_mejor = coste_x
                        iter_sin_mejora = 0
                        
                        tiempo_actual = time.time() - start_time
                        porcentaje_k = vecindarios_k[k] * 100
                        

                        historial_convergencia.append({
                            'Iteracion': iteracion_total,
                            'Tiempo_s': round(tiempo_actual, 2),
                            'FO_Mejor': round(coste_mejor, 2),
                            'Shake_%': porcentaje_k
                        })
                        
                        if logger: logger.info(f"      -> Iter {iteracion_total:03d} [{tiempo_actual:05.1f}s]: ¡Nueva FO Global: {coste_mejor:.2f} km! (Shake: {porcentaje_k}%)")
                    k = 0
                else:
                    k += 1
            else:
                k += 1
                
        iter_sin_mejora += 1
        
    historial_convergencia.append({
        'Iteracion': iteracion_total, 'Tiempo_s': round(time.time() - start_time, 2), 'FO_Mejor': round(coste_mejor, 2), 'Shake_%': 0.0
    })
        
    if logger: logger.info(f"   [VNS] Fin. Total Iters: {iteracion_total} | Mejor FO: {coste_mejor:.2f} km | Causa: {'Tiempo' if (time.time() - start_time) >= time_limit_sec else 'Estancamiento'}")
    
    return x_mejor, historial_convergencia