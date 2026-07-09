"""
Orquestador: BENCHMARK (GRASP)
"""
import os
import time
import pandas as pd
from Heurístico.src.data_pipeline import procesar_demanda_estocastica
from Heurístico.src.routing_osrm import generar_matrices_osrm
from Heurístico.src.grasp_constructor import construir_solucion_grasp
from Heurístico.src.logger import setup_logger

DATA_RAW_DIR = "./data/raw/"
OUTPUTS_DIR = "./outputs/results/"
RECOGIDAS_PATH = os.path.join(DATA_RAW_DIR, "data_03mayo25_16abril26.csv")
INVENTARIO_PATH = os.path.join(DATA_RAW_DIR, "inventario.csv")
DEPOT_LAT, DEPOT_LON = 28.11714256, -16.47846232

def run_heuristic_grasp():
    logger = setup_logger("GRASP_Benchmark")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    logger.info("="*60)
    logger.info(" BENCHMARK FASE 1: GRASP")
    logger.info("="*60)
    
    df_demanda = procesar_demanda_estocastica(RECOGIDAS_PATH, INVENTARIO_PATH)
    logger.info("Descargando/Procesando matrices OSRM...")
    df_dist, df_time = generar_matrices_osrm(df_demanda, DEPOT_LAT, DEPOT_LON)
    
    FLOTA_MAXIMA = 6
    ALPHA_GRASP = 0.5
    N_SEMILLAS = 50
    SEMILLAS = [2026 + i for i in range(N_SEMILLAS)] # Semillas deterministas secuenciales
    
    resultados_semillas = []
    mejor_solucion_global = None
    mejor_coste_global = float('inf')
    
    logger.info(f"Lanzando {len(SEMILLAS)} ejecuciones con Alpha={ALPHA_GRASP}")
    
    for i, semilla in enumerate(SEMILLAS):
        logger.info(f"-> Ejecución {i+1}/{len(SEMILLAS)} [Semilla: {semilla}]...")
        start_time = time.time()
        
        solucion = construir_solucion_grasp(
            df_dist=df_dist, 
            df_time=df_time, 
            df_demanda=df_demanda, 
            num_vehiculos=FLOTA_MAXIMA,
            alpha=ALPHA_GRASP,
            semilla=semilla
        )
        
        runtime = time.time() - start_time
        coste = solucion.coste_total_km
        nodos_huerfanos = len(solucion.nodos_no_asignados)
        
        resultados_semillas.append({
            'Semilla': semilla,
            'Runtime_s': runtime,
            'Coste_FO_Km': coste,
            'Huerfanos': nodos_huerfanos
        })
        
        # Guardar la mejor de todas para exportarla al final
        if coste < mejor_coste_global and nodos_huerfanos == 0:
            mejor_coste_global = coste
            mejor_solucion_global = solucion
            
    # Resumen Estadístico
    df_stats = pd.DataFrame(resultados_semillas)
    logger.info("="*60)
    logger.info(" RESUMEN ESTADÍSTICO (Multi-ejecución)")
    logger.info("="*60)
    logger.info(f" -> FO Mínima (Mejor):  {df_stats['Coste_FO_Km'].min():.2f} km")
    logger.info(f" -> FO Media (Promedio): {df_stats['Coste_FO_Km'].mean():.2f} km")
    logger.info(f" -> FO Desv. Estándar:  {df_stats['Coste_FO_Km'].std():.2f} km")
    logger.info(f" -> Tiempo Medio Ejec.: {df_stats['Runtime_s'].mean():.2f} s")
    
    df_stats.to_csv(os.path.join(OUTPUTS_DIR, f"benchmark_grasp_{N_SEMILLAS}_iterations.csv"), index=False)

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
        pd.DataFrame(rutas_export).to_csv(os.path.join(OUTPUTS_DIR, "benchmark_grasp_best.csv"), index=False)
        logger.info("Mejor ruta exportada a 'benchmark_grasp_best.csv'.")

if __name__ == "__main__":
    run_heuristic_grasp()