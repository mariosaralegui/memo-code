"""
COMPARACIÓN DE ESTRATEGIAS VND
"""
import os
import time
import copy
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

def run_compare_strategies():
    logger = setup_logger("VND_Compare")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    logger.info("="*60)
    logger.info(" INICIANDO COMPARATIVA DE ESTRATEGIAS VND")
    logger.info("="*60)
    
    df_demanda = procesar_demanda_estocastica(RECOGIDAS_PATH, INVENTARIO_PATH)
    logger.info("Descargando/Procesando matrices OSRM...")
    df_dist, df_time = generar_matrices_osrm(df_demanda, DEPOT_LAT, DEPOT_LON)
    

    # -------------------------------------------------------------

    N_SEMILLAS = 20
    SEMILLAS = [2026 + i for i in range(N_SEMILLAS)] # Semillas deterministas secuenciales
    ALPHA_GRASP = 0.8
    FLOTA_MAXIMA = 6
    
    # Define las estrategias que quieres enfrentar
    ESTRATEGIAS = [
        #{'nombre': 'Solo 2-opt',         'params': {'use_2opt': True, 'max_or_opt_k': 0, 'use_swap': False}},
        #{'nombre': '2-opt + OrOpt(k=1)','params': {'use_2opt': True, 'max_or_opt_k': 1, 'use_swap': False}},
        #{'nombre': '2-opt + OrOpt(k=2)','params': {'use_2opt': True, 'max_or_opt_k': 2, 'use_swap': False}},
        {'nombre': '2-opt + OrOpt(k=3)','params': {'use_2opt': True, 'max_or_opt_k': 3, 'use_swap': False}},
        #{'nombre': '2-opt + OrOpt(k=4)','params': {'use_2opt': True, 'max_or_opt_k': 4, 'use_swap': False}},
        #{'nombre': '2-opt + OrOpt(k=5)','params': {'use_2opt': True, 'max_or_opt_k': 5, 'use_swap': False}},
        #{'nombre': '2-opt + OrOpt(k=6)','params': {'use_2opt': True, 'max_or_opt_k': 6, 'use_swap': False}},
        #{'nombre': '2-opt + OrOpt(k=7)','params': {'use_2opt': True, 'max_or_opt_k': 7, 'use_swap': False}},
        #{'nombre': 'Full (k=1)', 'params': {'use_2opt': True, 'max_or_opt_k': 1, 'use_swap': True}},
        {'nombre': 'Full (k=3)', 'params': {'use_2opt': True, 'max_or_opt_k': 3, 'use_swap': True}}
        #{'nombre': 'Full (k=5)', 'params': {'use_2opt': True, 'max_or_opt_k': 5, 'use_swap': True}},
        #{'nombre': 'Full (k=7)', 'params': {'use_2opt': True, 'max_or_opt_k': 7, 'use_swap': True}}

    ]
    
    resultados = []
    
    for i, semilla in enumerate(SEMILLAS):
        logger.info(f"\n---> Evaluando Semilla {semilla} ({i+1}/{len(SEMILLAS)})")
        
        # Generar solución base
        solucion_base = construir_solucion_grasp(df_dist, df_time, df_demanda, FLOTA_MAXIMA, ALPHA_GRASP, semilla)
        fo_grasp = solucion_base.coste_total_km
        logger.info(f"   [GRASP Base] FO: {fo_grasp:.2f} km")
        
        fila_resultado = {'Semilla': semilla, 'FO_Base_GRASP': fo_grasp}
        
        # Evaluar cada estrategia sobre una copia limpia de la solución base
        for est in ESTRATEGIAS:
            sol_clon = copy.deepcopy(solucion_base)
            t0 = time.time()
            sol_vnd = run_vnd_global(sol_clon, **est['params'])
            t_vnd = time.time() - t0
            
            fo_vnd = sol_vnd.coste_total_km
            pct_mejora = ((fo_grasp - fo_vnd) / fo_grasp) * 100
            
            logger.info(f"   [{est['nombre']}] FO: {fo_vnd:.2f} km | Mejora: {pct_mejora:.2f}% | T_Extra: {t_vnd:.2f}s")
            
            fila_resultado[f"FO_{est['nombre']}"] = fo_vnd
            fila_resultado[f"Mejora_{est['nombre']}_%"] = pct_mejora
            fila_resultado[f"Tiempo_{est['nombre']}_s"] = t_vnd
            
        resultados.append(fila_resultado)
        
    pd.DataFrame(resultados).to_csv(os.path.join(OUTPUTS_DIR, "vnd_strategies.csv"), index=False)
    logger.info("\nComparativa completada. Archivo guardado: vnd_strategies.csv")

if __name__ == "__main__":
    run_compare_strategies()