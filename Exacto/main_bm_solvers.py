"""
Orquestador Unificado: Benchmark de Escalabilidad Exacta (Gurobi vs CP-SAT)
"""

import os
import time
import pandas as pd
from src.data_pipeline import procesar_demanda_estocastica
from src.routing_osrm import generar_matrices_osrm

# Importamos ambos solvers
from src.solver_exact_gurobi import resolver_pcvrp_benchmark
from src.solver_exact_cpsat import resolver_pcvrp_cpsat 
from src.logger import setup_logger

DATA_RAW_DIR = "./data/raw/"
OUTPUTS_DIR = "./outputs/results/"
RECOGIDAS_PATH = os.path.join(DATA_RAW_DIR, "data_03mayo25_16abril26.csv")
INVENTARIO_PATH = os.path.join(DATA_RAW_DIR, "inventario.csv")
DEPOT_LAT, DEPOT_LON = 28.11714256, -16.47846232

def run_unified_benchmark():
    logger = setup_logger("Exact_BM")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    logger.info("="*60)
    logger.info(" INICIANDO BENCHMARK UNIFICADO: GUROBI vs CP-SAT")
    logger.info("="*60)
    
    df_demanda_completa = procesar_demanda_estocastica(RECOGIDAS_PATH, INVENTARIO_PATH)
    


    #-----------------------------------------------------------------
    # Saltos para ver el muro exponencial
    tamanos_n = [20, 30, 40, 50, 60, 80]
    tiempo_limite = 3600  # 1 Hora máximo por intento
    #-----------------------------------------------------------------



    resultados_totales = []
    csv_path = os.path.join(OUTPUTS_DIR, "benchmark_exact_scalability.csv")

    for n in tamanos_n:
        logger.info(f"\n" + "-"*40)
        logger.info(f" PREPARANDO INSTANCIA N = {n} CONTENEDORES")
        logger.info("-"*40)
        
        # Mismo random_state para que ambos solvers jueguen en el mismo tablero
        df_instancia = df_demanda_completa.sample(n=n, random_state=2026).copy()
        df_dist, df_time = generar_matrices_osrm(df_instancia, DEPOT_LAT, DEPOT_LON)
        

        vehiculos_necesarios = 6
        
        # -------------------------------------------------------------
        # 1. EJECUCIÓN GUROBI
        # -------------------------------------------------------------
        logger.info(f"[GUROBI] Iniciando resolución (Max {tiempo_limite}s)...")
        try:
            res_gurobi = resolver_pcvrp_benchmark(
                df_demanda=df_instancia, 
                df_dist=df_dist, 
                df_time=df_time, 
                num_vehiculos=vehiculos_necesarios, 
                time_limit=tiempo_limite, 
                mip_gap=0.01
            )
            resultados_totales.append({
                'Solver': 'Gurobi',
                'N_Contenedores': n,
                'Status': res_gurobi.get('Status', 'ERROR'),
                'Tiempo_s': res_gurobi.get('Runtime_Sec', -1),
                'FO_Km': res_gurobi.get('FO_Km', -1)
            })
            logger.info(f"[GUROBI] Fin: {res_gurobi.get('Status')} | {res_gurobi.get('Runtime_Sec'):.2f}s | FO: {res_gurobi.get('FO_Km')}")
        except Exception as e:
            logger.error(f"[GUROBI] Fallo por error o falta de memoria: {e}")
            resultados_totales.append({'Solver': 'Gurobi', 'N_Contenedores': n, 'Status': 'OOM_OR_ERROR', 'Tiempo_s': tiempo_limite, 'FO_Km': None})

        # Guardado de seguridad iterativo
        pd.DataFrame(resultados_totales).to_csv(csv_path, index=False)

        # -------------------------------------------------------------
        # 2. EJECUCIÓN CP-SAT
        # -------------------------------------------------------------
        logger.info(f"[CP-SAT] Iniciando resolución (Max {tiempo_limite}s)...")
        try:
            res_cpsat, _ = resolver_pcvrp_cpsat(
                df_demanda=df_instancia, 
                df_dist=df_dist, 
                df_time=df_time, 
                num_vehiculos=vehiculos_necesarios, 
                time_limit=tiempo_limite
            )
            resultados_totales.append({
                'Solver': 'CP-SAT',
                'N_Contenedores': n,
                'Status': res_cpsat.get('Status', 'ERROR'),
                'Tiempo_s': res_cpsat.get('Runtime_Sec', -1),
                'FO_Km': res_cpsat.get('FO_Km', -1)
            })
            logger.info(f"[CP-SAT] Fin: {res_cpsat.get('Status')} | {res_cpsat.get('Runtime_Sec'):.2f}s | FO: {res_cpsat.get('FO_Km')}")
        except Exception as e:
            logger.error(f"[CP-SAT] Fallo por error o falta de memoria: {e}")
            resultados_totales.append({'Solver': 'CP-SAT', 'N_Contenedores': n, 'Status': 'OOM_OR_ERROR', 'Tiempo_s': tiempo_limite, 'FO_Km': None})

        # Guardado de seguridad iterativo
        pd.DataFrame(resultados_totales).to_csv(csv_path, index=False)

    logger.info("="*60)
    logger.info(" BENCHMARK UNIFICADO FINALIZADO CON ÉXITO")
    logger.info(f" Resultados guardados en: {csv_path}")

if __name__ == "__main__":
    run_unified_benchmark()