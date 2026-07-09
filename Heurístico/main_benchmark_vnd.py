"""
Orquestador: BENCHMARK FASE 2 (GRASP -> VND)
"""
import os
import time
import pandas as pd
from Heurístico.src.data_pipeline import procesar_demanda_estocastica
from Heurístico.src.routing_osrm import generar_matrices_osrm
from Heurístico.src.grasp_constructor import construir_solucion_grasp
from Heurístico.src.vnd_framework import run_vnd_global
from Heurístico.src.logger import setup_logger

DATA_RAW_DIR = "./data/raw/"
OUTPUTS_DIR = "./outputs/results/"
RECOGIDAS_PATH = os.path.join(DATA_RAW_DIR, "data_03mayo25_16abril26.csv")
INVENTARIO_PATH = os.path.join(DATA_RAW_DIR, "inventario.csv")
DEPOT_LAT, DEPOT_LON = 28.11714256, -16.47846232

def run_benchmark_vnd():
    logger = setup_logger("VND_Benchmark")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    logger.info("="*60)
    logger.info(" BENCHMARK FASE 2: GRASP -> VND")
    logger.info("="*60)
    
    df_demanda = procesar_demanda_estocastica(RECOGIDAS_PATH, INVENTARIO_PATH)
    logger.info("Descargando/Procesando matrices OSRM...")
    df_dist, df_time = generar_matrices_osrm(df_demanda, DEPOT_LAT, DEPOT_LON)
    
    # -------------------------------------------------------------
    # PARÁMETROS
    # -------------------------------------------------------------
    N_SEMILLAS = 50
    SEMILLAS = [2026 + i for i in range(N_SEMILLAS)]
    FLOTA_MAXIMA = 6
    ALPHA_GRASP = 0.8 # Óptimo extraído del estudio reactivo
    
    # CONFIGURACIÓN DEL VND ELEGIDA 
    USE_2OPT = True
    MAX_OR_OPT_K = 3
    USE_SWAP = True
    
    logger.info(f"Configuración VND -> 2opt: {USE_2OPT} | OrOpt Max k: {MAX_OR_OPT_K} | Swap: {USE_SWAP}")
    logger.info(f"Lanzando {N_SEMILLAS} ejecuciones...")
    
    resultados = []
    mejor_solucion_global = None
    mejor_coste_vnd = float('inf')
    
    for i, semilla in enumerate(SEMILLAS):
        t_start = time.time()
        
        # 1. GRASP
        solucion_grasp = construir_solucion_grasp(df_dist, df_time, df_demanda, FLOTA_MAXIMA, ALPHA_GRASP, semilla)
        t_grasp = time.time()
        fo_grasp = solucion_grasp.coste_total_km
        
        # 2. VND
        solucion_vnd = run_vnd_global(solucion_grasp, USE_2OPT, MAX_OR_OPT_K, USE_SWAP)
        t_vnd = time.time()
        fo_vnd = solucion_vnd.coste_total_km
        
        # Cálculo de Métricas
        tiempo_grasp_s = t_grasp - t_start
        tiempo_vnd_s = t_vnd - t_grasp
        mejora_pct = ((fo_grasp - fo_vnd) / fo_grasp) * 100
        
        logger.info(f" [{i+1}/{N_SEMILLAS}] FO GRASP: {fo_grasp:.2f} | FO VND: {fo_vnd:.2f} (-{mejora_pct:.2f}%)")
        
        resultados.append({
            'Semilla': semilla,
            'FO_GRASP': fo_grasp,
            'FO_VND': fo_vnd,
            'Mejora_%': mejora_pct,
            'Tiempo_GRASP_s': tiempo_grasp_s,
            'Tiempo_VND_s': tiempo_vnd_s
        })
        
        # Guardar el Campeón Global
        if fo_vnd < mejor_coste_vnd and len(solucion_vnd.nodos_no_asignados) == 0:
            mejor_coste_vnd = fo_vnd
            mejor_solucion_global = solucion_vnd

    # Resumen y Estadísticas Finales
    df_stats = pd.DataFrame(resultados)
    logger.info("\n" + "="*60)
    logger.info(f" RESUMEN ESTADÍSTICO DEL BENCHMARK ({N_SEMILLAS} Semillas)")
    logger.info("="*60)
    logger.info(f" -> FO VND Mínima (Mejor):   {df_stats['FO_VND'].min():.2f} km")
    logger.info(f" -> FO VND Media (Promedio): {df_stats['FO_VND'].mean():.2f} km")
    logger.info(f" -> Mejora Media vs GRASP:   {df_stats['Mejora_%'].mean():.2f} %")
    logger.info(f" -> T. Medio Total (Ejec.):  {(df_stats['Tiempo_GRASP_s'] + df_stats['Tiempo_VND_s']).mean():.2f} s")
    
    df_stats.to_csv(os.path.join(OUTPUTS_DIR, f"benchmark_vnd_{N_SEMILLAS}_iterations.csv"), index=False)
    
    # Exportar la ruta ganadora absoluta
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
        pd.DataFrame(rutas_export).to_csv(os.path.join(OUTPUTS_DIR, "benchmark_vnd_best.csv"), index=False)
        logger.info("Mejor ruta exportada a 'benchmark_vnd_best.csv'")

if __name__ == "__main__":
    run_benchmark_vnd()