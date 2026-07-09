"""
Modulo 1: Preprocesamiento de Datos y Safety Stock Estocastico (MAD)
"""
import pandas as pd
import numpy as np

def calcular_mad(x: pd.Series) -> float:
    """Calcula la Desviacion Absoluta de la Mediana (MAD) ignorando NaNs."""
    if len(x) == 0:
        return 0.0
    mediana = x.median()
    return (x - mediana).abs().median()

def procesar_demanda_estocastica(recogidas_path: str, inventario_path: str, z_score: float = 1.0) -> pd.DataFrame:
    print("[Pipeline] Iniciando análisis de intervalos temporales y Safety Stock Robusto (MAD)...")
    
    # Carga y limpieza
    df = pd.read_csv(recogidas_path, low_memory=False)
    df = df.dropna(subset=['Cod. Contenedor', 'Fecha']).copy()
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Fecha']).sort_values(by=['Cod. Contenedor', 'Fecha'])

    # Calculo de intervalos
    df['Dias_Desde_Ultima'] = df.groupby('Cod. Contenedor')['Fecha'].diff().dt.total_seconds() / 86400.0
    df['Dias_Desde_Ultima'] = df['Dias_Desde_Ultima'].fillna(2.0).clip(lower=0.5)

    # Filtrado de visitas validas
    df_valid = df[df['Tipo inc'].isna()].copy()
    df_valid['Tasa_Local_Kg_Dia'] = df_valid['Peso'] / df_valid['Dias_Desde_Ultima']

    # Capacidad practica (P95) y umbral seguro (90%)
    pesos_positivos = df_valid.loc[df_valid['Peso'] > 0, 'Peso']
    capacidad_practica = np.percentile(pesos_positivos, 95) if not pesos_positivos.empty else 135.0
    umbral_seguro_kg = capacidad_practica * 0.90

    # Agregacion estadistica robusta por contenedor (Uso de MAD)
    stats = df_valid.groupby('Cod. Contenedor').agg(
        Tasa_Generacion_Diaria_Kg=('Tasa_Local_Kg_Dia', 'median'), 
        MAD_Tasa_Kg_Dia=('Tasa_Local_Kg_Dia', calcular_mad),
        Visitas_Validas=('Fecha', 'count')
    ).reset_index()

    mediana_mad_global = stats['MAD_Tasa_Kg_Dia'].median()
    stats['MAD_Tasa_Kg_Dia'] = stats['MAD_Tasa_Kg_Dia'].fillna(mediana_mad_global)

    # Calculo Estocastico (Demanda Pesimista Robusta)
    stats['Tasa_Pesimista_Kg_Dia'] = stats['Tasa_Generacion_Diaria_Kg'] + (z_score * stats['MAD_Tasa_Kg_Dia'])
    stats['Umbral_Recogida_Efectivo'] = umbral_seguro_kg

    # Asignacion de frecuencias
    stats['Dias_Para_Umbral'] = np.where(
        stats['Tasa_Pesimista_Kg_Dia'] > 0,
        stats['Umbral_Recogida_Efectivo'] / stats['Tasa_Pesimista_Kg_Dia'],
        999
    )
    
    def asignar_frecuencia(dias):
        if dias >= 7: return 1
        elif dias >= 3: return 2
        else: return 3

    stats['Opt_Frecuencia_Semanal'] = stats['Dias_Para_Umbral'].apply(asignar_frecuencia)

    # Integracion espacial
    inv = pd.read_csv(inventario_path, low_memory=False)
    inv = inv[['Código', 'Latitud', 'Longitud']].rename(columns={'Código': 'Cod. Contenedor'})
    df_final = pd.merge(stats, inv, on='Cod. Contenedor', how='inner').dropna(subset=['Latitud', 'Longitud'])
    
    print(f"[Pipeline] Completado. N={len(df_final)}. Umbral efectivo: {umbral_seguro_kg:.1f} kg.")
    return df_final