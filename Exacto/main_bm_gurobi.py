"""
Orquestador: Generación de Curva de Complejidad (Fase 1 - Gurobi)
"""
import os
import pandas as pd
from src.data_pipeline import procesar_demanda_estocastica
from src.routing_osrm import generar_matrices_osrm
from src.solver_exact_gurobi import resolver_pcvrp_benchmark
from src.logger import setup_logger

DATA_RAW_DIR = "./data/raw/"
OUTPUTS_DIR = "./outputs/results/"
RECOGIDAS_PATH = os.path.join(DATA_RAW_DIR, "data_03mayo25_16abril26.csv")
INVENTARIO_PATH = os.path.join(DATA_RAW_DIR, "inventario.csv")
DEPOT_LAT, DEPOT_LON = 28.11714256, -16.47846232

def run_benchmark():
    logger = setup_logger("Gurobi_BM")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    logger.info("="*60)
    logger.info(" INICIANDO EXPERIMENTO BENCHMARK (MILP GUROBI)")
    logger.info("="*60)
    
    df_demanda_completa = procesar_demanda_estocastica(RECOGIDAS_PATH, INVENTARIO_PATH)
    


    #------------------------------------------------------------------------------
    tamaños_n = [10, 309]
    tiempo_limite_gurobi = 1800  
    gap = 0.01
    #------------------------------------------------------------------------------



    resultados_benchmark = []

    for n in tamaños_n:
        logger.info(f"\n---> Evaluando Instancia de N={n} nodos...")
        df_instancia = df_demanda_completa.sample(n=n, random_state=123).copy()
        
        df_dist, df_time = generar_matrices_osrm(df_instancia, DEPOT_LAT, DEPOT_LON)
        vehiculos_necesarios = max(2, int(n / 15) + 1)
        
        metricas = resolver_pcvrp_benchmark(
            df_demanda=df_instancia, 
            df_dist=df_dist, 
            df_time=df_time, 
            num_vehiculos=vehiculos_necesarios, 
            time_limit=tiempo_limite_gurobi, 
            mip_gap=gap
        )
        
        metricas['N_Nodos'] = n
        resultados_benchmark.append(metricas)
        logger.info(f"Resultado N={n}: Estado={metricas['Status']} | Tiempo={metricas['Runtime_Sec']}s | Gap={metricas['MIPGap']}")
        
        df_resultados = pd.DataFrame(resultados_benchmark)
        df_resultados = df_resultados[['N_Nodos', 'Status', 'Runtime_Sec', 'ObjVal', 'MIPGap']]
        df_resultados.to_csv(os.path.join(OUTPUTS_DIR, "benchmark_results_gurobi.csv"), index=False)

    logger.info("\n" + "="*60)
    logger.info(f" BENCHMARK GUROBI COMPLETADO. Resultados guardados en {OUTPUTS_DIR}")
    logger.info("="*60)

if __name__ == "__main__":
    run_benchmark()