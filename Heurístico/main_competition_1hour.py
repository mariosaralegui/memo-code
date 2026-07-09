"""
Experimento de Presupuesto Computacional (1 Hora)
Multi-start GRASP+VND (Exploración) vs GRASP+VND+VNS (Explotación)
"""
import os
import time
import pandas as pd
import logging
from Heurístico.src.data_pipeline import procesar_demanda_estocastica
from Heurístico.src.routing_osrm import generar_matrices_osrm
from Heurístico.src.grasp_constructor import construir_solucion_grasp
from Heurístico.src.vnd_framework import run_vnd_global
from Heurístico.src.vns_framework import run_vns_global
from Heurístico.src.logger import setup_logger

logging.raiseExceptions = False

DATA_RAW_DIR = "./data/raw/"
OUTPUTS_DIR = "./outputs/results/"
RECOGIDAS_PATH = os.path.join(DATA_RAW_DIR, "data_03mayo25_16abril26.csv")
INVENTARIO_PATH = os.path.join(DATA_RAW_DIR, "inventario.csv")
DEPOT_LAT, DEPOT_LON = 28.11714256, -16.47846232

# PARÁMETROS DEL EXPERIMENTO
TIEMPO_LIMITE_SEC = 3600  # 1 hora = 3600 segundos
N_SESIONES = 3            # 3 sesiones independientes (que 6 horas son ya muchas)
NUM_VEHICULOS = 6 
ALPHA_GRASP = 0.8
VND_PARAMS = {'use_2opt': True, 'max_or_opt_k': 3, 'use_swap': True}

def run_competition():
    logger = setup_logger("Competition_1h")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    logger.info("="*70)
    logger.info(f" INICIANDO COMPETICIÓN DE PRESUPUESTO FIJO: {TIEMPO_LIMITE_SEC}s")
    logger.info("="*70)
    
    df_demanda = procesar_demanda_estocastica(RECOGIDAS_PATH, INVENTARIO_PATH)
    df_dist, df_time = generar_matrices_osrm(df_demanda, DEPOT_LAT, DEPOT_LON)
    
    historial_global = []

    for sesion in range(1, N_SESIONES + 1):
        semilla_base = 2026 + (sesion * 10) 
        logger.info(f"\n--- INICIANDO SESIÓN {sesion}/{N_SESIONES} (Semilla Base: {semilla_base}) ---")

        # =====================================================================
        # ENFOQUE A: Multi-start GRASP + VND (Exploración)
        # =====================================================================
        logger.info("[Enfoque A] Arrancando Multi-start GRASP+VND...")
        start_time_A = time.time()
        mejor_fo_A = float('inf')
        iteracion_A = 0
        
        historial_global.append({'Enfoque': 'A_MultiStart_VND', 'Sesion': sesion, 'Tiempo_s': 0.0, 'FO_Mejor': None})

        while (time.time() - start_time_A) < TIEMPO_LIMITE_SEC:
            semilla_actual = semilla_base + iteracion_A
            
            sol_grasp = construir_solucion_grasp(df_dist, df_time, df_demanda, NUM_VEHICULOS, ALPHA_GRASP, semilla_actual)
            # --- FIX: Pasamos VND_PARAMS ---
            sol_vnd = run_vnd_global(sol_grasp, **VND_PARAMS) 
            
            coste_actual = sol_vnd.coste_total_km # Usamos la variable directa
            tiempo_transcurrido = time.time() - start_time_A
            
            if coste_actual < mejor_fo_A:
                mejor_fo_A = coste_actual
                logger.info(f"  [A] Mejor FO encontrada: {mejor_fo_A:.2f} (Iter: {iteracion_A}, t={tiempo_transcurrido:.1f}s)")
            
            historial_global.append({
                'Enfoque': 'A_MultiStart_VND',
                'Sesion': sesion,
                'Tiempo_s': round(tiempo_transcurrido, 2),
                'FO_Mejor': round(mejor_fo_A, 2)
            })
            iteracion_A += 1

        # =====================================================================
        # ENFOQUE B: GRASP + VND + VNS Profundo (Explotación)
        # =====================================================================
        logger.info(f"\n[Enfoque B] Arrancando VNS Profundo...")
        start_time_B = time.time()
        
        historial_global.append({'Enfoque': 'B_VNS_Profundo', 'Sesion': sesion, 'Tiempo_s': 0.0, 'FO_Mejor': None})

        sol_grasp_B = construir_solucion_grasp(df_dist, df_time, df_demanda, NUM_VEHICULOS, ALPHA_GRASP, semilla_base)
        sol_vnd_B = run_vnd_global(sol_grasp_B, **VND_PARAMS)
        
        tiempo_consumido_inicial = time.time() - start_time_B
        mejor_fo_B = sol_vnd_B.coste_total_km
        
        historial_global.append({
            'Enfoque': 'B_VNS_Profundo',
            'Sesion': sesion,
            'Tiempo_s': round(tiempo_consumido_inicial, 2),
            'FO_Mejor': round(mejor_fo_B, 2)
        })

        tiempo_restante_vns = TIEMPO_LIMITE_SEC - tiempo_consumido_inicial
        logger.info(f"  [B] FO Inicial (Tras VND): {mejor_fo_B:.2f}. Tiempo restante VNS: {tiempo_restante_vns:.1f}s")
        

        sol_final_B, historial_vns = run_vns_global(
            solucion_inicial=sol_vnd_B, 
            df_demanda=df_demanda, 
            time_limit_sec=tiempo_restante_vns, 
            vnd_params=VND_PARAMS, 
            max_iter_sin_mejora=999999, # Infinito, para que solo corte por tiempo
            logger=logger
        )

        for registro in historial_vns:
            historial_global.append({
                'Enfoque': 'B_VNS_Profundo',
                'Sesion': sesion,
                'Tiempo_s': round(registro['Tiempo_s'] + tiempo_consumido_inicial, 2),
                'FO_Mejor': registro['FO_Mejor']
            })

    # =====================================================================
    # EXPORTACIÓN
    # =====================================================================
    df_resultados = pd.DataFrame(historial_global)
    df_resultados['FO_Mejor'] = df_resultados.groupby(['Enfoque', 'Sesion'])['FO_Mejor'].bfill()
    
    csv_path = os.path.join(OUTPUTS_DIR, "competition_results_1h.csv")
    df_resultados.to_csv(csv_path, index=False)
    logger.info(f"\n¡Experimento finalizado! Datos guardados en: {csv_path}")

if __name__ == "__main__":
    run_competition()