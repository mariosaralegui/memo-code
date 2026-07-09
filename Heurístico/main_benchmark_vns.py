"""
Orquestador: BENCHMARK FASE 3 (GRASP -> VND -> VNS)
"""
import os
import time
import pandas as pd
from Heurístico.src.data_pipeline import procesar_demanda_estocastica
from Heurístico.src.routing_osrm import generar_matrices_osrm
from Heurístico.src.grasp_constructor import construir_solucion_grasp
from Heurístico.src.vnd_framework import run_vnd_global
from Heurístico.src.vns_framework import run_vns_global
from Heurístico.src.logger import setup_logger

import logging
logging.raiseExceptions = False

DATA_RAW_DIR = "./data/raw/"
OUTPUTS_DIR = "./outputs/results/"
RECOGIDAS_PATH = os.path.join(DATA_RAW_DIR, "data_03mayo25_16abril26.csv")
INVENTARIO_PATH = os.path.join(DATA_RAW_DIR, "inventario.csv")
DEPOT_LAT, DEPOT_LON = 28.11714256, -16.47846232

def run_benchmark_vns():
    logger = setup_logger("VNS_Benchmark")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    logger.info("="*60)
    logger.info(" BENCHMARK FASE 3 (CON REGISTRO DE CONVERGENCIA)")
    logger.info("="*60)
    
    df_demanda = procesar_demanda_estocastica(RECOGIDAS_PATH, INVENTARIO_PATH)
    logger.info("Descargando/Procesando matrices OSRM...")
    df_dist, df_time = generar_matrices_osrm(df_demanda, DEPOT_LAT, DEPOT_LON)
    
    # -----------------------------------------------------------------------------
    N_SEMILLAS = 20 
    SEMILLAS = [2026 + i for i in range(N_SEMILLAS)]
    FLOTA_MAXIMA = 6
    ALPHA_GRASP = 0.8 
    
    VND_PARAMS = {'use_2opt': True, 'max_or_opt_k': 3, 'use_swap': True}
    
    TIEMPO_LIMITE_VNS = 1800       # 30 minutos por semilla
    MAX_ITER_SIN_MEJORA = 25      
    # -----------------------------------------------------------------------------

    logger.info(f"Lanzando {N_SEMILLAS} semillas.")
    
    resultados = []
    datos_convergencia_global = [] 
    mejor_solucion_global = None
    mejor_coste_vns = float('inf')
    
    for i, semilla in enumerate(SEMILLAS):
        logger.info(f"\n---> Evaluando Semilla {semilla} ({i+1}/{N_SEMILLAS})")
        t_start = time.time()
        
        # 1. GRASP
        solucion_grasp = construir_solucion_grasp(df_dist, df_time, df_demanda, FLOTA_MAXIMA, ALPHA_GRASP, semilla)
        t_grasp = time.time()
        fo_grasp = solucion_grasp.coste_total_km
        logger.info(f"   [GRASP] FO: {fo_grasp:.2f} km")
        
        # 2. VND
        solucion_vnd = run_vnd_global(solucion_grasp, **VND_PARAMS)
        t_vnd = time.time()
        fo_vnd = solucion_vnd.coste_total_km
        logger.info(f"   [VND]   FO: {fo_vnd:.2f} km")
        
        # 3. VNS (Ahora recibe la tupla)
        solucion_vns, historial_vns = run_vns_global(
            solucion_vnd, 
            df_demanda, 
            TIEMPO_LIMITE_VNS, 
            VND_PARAMS, 
            max_iter_sin_mejora=MAX_ITER_SIN_MEJORA, 
            logger=logger
        )
        t_vns = time.time()
        fo_vns = solucion_vns.coste_total_km
        

        for registro in historial_vns:
            registro['Semilla'] = semilla
            datos_convergencia_global.append(registro)
            
        mejora_total_pct = ((fo_grasp - fo_vns) / fo_grasp) * 100
        logger.info(f"   [FINAL] FO: {fo_vns:.2f} km (Mejora desde GRASP: {mejora_total_pct:.2f}%)")
        
        resultados.append({
            'Semilla': semilla,
            'FO_GRASP': fo_grasp,
            'FO_VND': fo_vnd,
            'FO_VNS': fo_vns,
            'Mejora_Global_%': mejora_total_pct,
            'Tiempo_Total_s': t_vns - t_start
        })
        
        if fo_vns < mejor_coste_vns and len(solucion_vns.nodos_no_asignados) == 0:
            mejor_coste_vns = fo_vns
            mejor_solucion_global = solucion_vns

    # Exportaciones Finales
    df_stats = pd.DataFrame(resultados)
    df_stats.to_csv(os.path.join(OUTPUTS_DIR, f"benchmark_vns_{N_SEMILLAS}_iterations.csv"), index=False)
    

    df_conv = pd.DataFrame(datos_convergencia_global)
    df_conv = df_conv[['Semilla', 'Iteracion', 'Tiempo_s', 'FO_Mejor', 'Shake_%']] # Ordenar columnas
    df_conv.to_csv(os.path.join(OUTPUTS_DIR, "vns_convergence_data.csv"), index=False)
    logger.info("Archivo de convergencia exportado a 'vns_convergence_data.csv'")
    
    if mejor_solucion_global:
        rutas_export = []
        for dia in mejor_solucion_global.dias:
            for ruta in mejor_solucion_global.rutas[dia]:
                if len(ruta.nodos) > 2:
                    rutas_export.append({
                        'Dia': dia, 'Vehiculo_ID': f"{dia[:2]}-{ruta.id_vehiculo}",
                        'N_Contenedores': len(ruta.nodos) - 2,
                        'Carga_Kg': round(ruta.carga_actual, 1),
                        'Ocupacion_%': round((ruta.carga_actual / 10000.0) * 100, 1),
                        'Distancia_Km': round(ruta.distancia_km, 2),
                        'Tiempo_Horas': round(ruta.tiempo_h, 2),
                        'Ruta_Nodos': " -> ".join(ruta.nodos)
                    })
        pd.DataFrame(rutas_export).to_csv(os.path.join(OUTPUTS_DIR, "benchmark_vns_best.csv"), index=False)
        logger.info("Tabla de rutas exportada a 'benchmark_vns_best.csv'")

if __name__ == "__main__":
    run_benchmark_vns()


