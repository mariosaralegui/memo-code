"""
Orquestador: GRASP REACTIVO (Aprendizaje Estadístico de Hiperparámetros)
"""
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from Heurístico.src.data_pipeline import procesar_demanda_estocastica
from Heurístico.src.routing_osrm import generar_matrices_osrm
from Heurístico.src.grasp_constructor import construir_solucion_grasp
from Heurístico.src.logger import setup_logger

DATA_RAW_DIR = "./data/raw/"
OUTPUTS_DIR = "./outputs/results/"
RECOGIDAS_PATH = os.path.join(DATA_RAW_DIR, "data_03mayo25_16abril26.csv")
INVENTARIO_PATH = os.path.join(DATA_RAW_DIR, "inventario.csv")
DEPOT_LAT, DEPOT_LON = 28.11714256, -16.47846232

class ControladorReactivo:
    """Motor matemático para la actualización de probabilidades del GRASP Reactivo."""
    def __init__(self, alphas: list, iteraciones_calentamiento: int = 18, delta: float = 2.0):
        self.alphas = alphas
        self.probabilidades = {a: 1.0 / len(alphas) for a in alphas}
        self.historial = {a: [] for a in alphas}
        self.iteraciones_calentamiento = iteraciones_calentamiento
        self.delta = delta # Factor de amplificación de diferencias
        self.z_star = float('inf') # Mejor FO global
        self.iteracion_actual = 0

    def seleccionar_alpha(self) -> float:
        """Selecciona un alpha basado en la distribución de probabilidad actual."""
        claves = list(self.probabilidades.keys())
        probs = list(self.probabilidades.values())
        return np.random.choice(claves, p=probs)

    def registrar_resultado(self, alpha: float, coste: float):
        """Almacena el coste y recalcula las probabilidades si superó el calentamiento."""
        self.historial[alpha].append(coste)
        self.iteracion_actual += 1
        
        if coste < self.z_star:
            self.z_star = coste

        # Solo actualizamos probabilidades después del calentamiento y si todos los alphas han sido probados
        if self.iteracion_actual > self.iteraciones_calentamiento and all(len(v) > 0 for v in self.historial.values()):
            self._actualizar_probabilidades()

    def _actualizar_probabilidades(self):
        """Aplica la fórmula probabilística de Prais & Ribeiro para GRASP Reactivo."""
        q_values = {}
        for alpha in self.alphas:
            z_avg = np.mean(self.historial[alpha])
            # q_i = (z* / z_avg)^delta
            q_values[alpha] = (self.z_star / z_avg) ** self.delta

        suma_q = sum(q_values.values())
        
        for alpha in self.alphas:
            self.probabilidades[alpha] = q_values[alpha] / suma_q

    def obtener_resumen_probs(self) -> str:
        resumen = " | ".join([f"a={a}: {p*100:.1f}%" for a, p in self.probabilidades.items()])
        return f"[{resumen}]"


def run_grasp_reactivo():
    logger = setup_logger("GRASP_Reactivo")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    logger.info("="*60)
    logger.info(" INICIANDO GRASP REACTIVO (Aprendizaje de Alpha)")
    logger.info("="*60)
    
    df_demanda = procesar_demanda_estocastica(RECOGIDAS_PATH, INVENTARIO_PATH)
    logger.info("Descargando/Procesando matrices OSRM...")
    df_dist, df_time = generar_matrices_osrm(df_demanda, DEPOT_LAT, DEPOT_LON)
    
    # -------------------------------------------------------------
    # PARÁMETROS DEL EXPERIMENTO REACTIVO
    # -------------------------------------------------------------
    ITERACIONES = 100
    ALPHAS_CANDIDATOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    FLOTA_MAXIMA = 6
    SEMILLA_BASE = 2026
    
    controlador = ControladorReactivo(ALPHAS_CANDIDATOS, iteraciones_calentamiento=len(ALPHAS_CANDIDATOS)*2)
    
    logger.info(f"Lanzando {ITERACIONES} iteraciones. Alphas: {ALPHAS_CANDIDATOS}")
    
    resultados = []
    mejor_solucion_global = None
    mejor_coste_global = float('inf')
    
    for i in range(ITERACIONES):
        semilla_actual = SEMILLA_BASE + i
        alpha_elegido = controlador.seleccionar_alpha()
        
        t_start = time.time()
        solucion = construir_solucion_grasp(
            df_dist, df_time, df_demanda, FLOTA_MAXIMA, alpha_elegido, semilla_actual
        )
        t_runtime = time.time() - t_start
        
        coste = solucion.coste_total_km
        controlador.registrar_resultado(alpha_elegido, coste)
        
        # Loggear la evolución (Mostramos cómo van cambiando las probabilidades)
        if i % 5 == 0 or coste < mejor_coste_global:
            logger.info(f" -> Iter {i+1:03d} [a={alpha_elegido}] | FO: {coste:.2f} km | Probs: {controlador.obtener_resumen_probs()}")
            
        resultados.append({
            'Iteracion': i + 1,
            'Semilla': semilla_actual,
            'Alpha_Usado': alpha_elegido,
            'FO_Km': coste,
            'Tiempo_s': t_runtime
        })
        
        if coste < mejor_coste_global and len(solucion.nodos_no_asignados) == 0:
            mejor_coste_global = coste
            mejor_solucion_global = solucion

    # Exportar resultados
    df_stats = pd.DataFrame(resultados)
    df_stats.to_csv(os.path.join(OUTPUTS_DIR, "benchmark_grasp_reactivo.csv"), index=False)
    
    # Evaluar qué Alpha fue el ganador estadístico
    logger.info("\n" + "="*60)
    logger.info(" DISTRIBUCIÓN FINAL DE PROBABILIDADES APRENDIDAS")
    logger.info("="*60)
    for a in ALPHAS_CANDIDATOS:
        logger.info(f" Alpha {a}: {controlador.probabilidades[a]*100:.2f}% de preferencia")

if __name__ == "__main__":
    run_grasp_reactivo()