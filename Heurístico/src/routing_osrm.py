"""
Motor Topológico OSRM
"""
import requests
import time
import numpy as np
import pandas as pd
from typing import Tuple



OSRM_BASE_URL = "http://router.project-osrm.org/table/v1/driving/"

def generar_matrices_osrm(df_nodos: pd.DataFrame, depot_lat: float, depot_lon: float, chunk_size: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        df_nodos (pd.DataFrame): DataFrame con los nodos a incluir en las matrices.
        depot_lat (float): Latitud del depósito.
        depot_lon (float): Longitud del depósito.
        chunk_size (int): Tamaño de los bloques para dividir las peticiones a OSRM.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Matrices de distancia y tiempo.
    """

    nodos_ids = df_nodos['Cod. Contenedor'].tolist()
    coords = [(depot_lon, depot_lat)] 
    
    for _, row in df_nodos.iterrows():
        coords.append((row['Longitud'], row['Latitud']))
        
    etiquetas = ['PIRS'] + nodos_ids
    n = len(etiquetas)
    print(f"[OSRM] Construyendo matrices de {n}x{n} nodos...")
    
    matriz_dist = np.zeros((n, n))
    matriz_time = np.zeros((n, n))
    
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            chunk_i = list(range(i, min(i + chunk_size, n)))
            chunk_j = list(range(j, min(j + chunk_size, n)))
            indices_unicos = sorted(list(set(chunk_i + chunk_j)))

            if len(indices_unicos) < 2:
                continue

            coords_str = ";".join([f"{coords[idx][0]:.6f},{coords[idx][1]:.6f}" for idx in indices_unicos])
            mapa_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_unicos)}
            src_str = ";".join([str(mapa_local[idx]) for idx in chunk_i])
            dst_str = ";".join([str(mapa_local[idx]) for idx in chunk_j])
            
            url = f"{OSRM_BASE_URL}{coords_str}?sources={src_str}&destinations={dst_str}&annotations=distance,duration"
            
            exito = False
            for intento in range(3):
                try:
                    respuesta = requests.get(url, timeout=15)
                    if respuesta.status_code == 200:
                        data = respuesta.json()
                        dists = np.array(data['distances']) / 1000.0
                        durs = np.array(data['durations']) / 3600.0
                        
                        for idx_local_i, global_i in enumerate(chunk_i):
                            for idx_local_j, global_j in enumerate(chunk_j):
                                matriz_dist[global_i, global_j] = dists[idx_local_i, idx_local_j]
                                matriz_time[global_i, global_j] = durs[idx_local_i, idx_local_j]
                        exito = True
                        break
                    else:
                        raise ConnectionError(f"HTTP {respuesta.status_code}")
                except Exception as e:
                    time.sleep(2 ** (intento + 1))
            
            if not exito:
                raise RuntimeError(f"Fallo crítico OSRM en bloque [{i}-{j}].")
            time.sleep(1.0) 
            
    df_dist = pd.DataFrame(matriz_dist, index=etiquetas, columns=etiquetas)
    df_time = pd.DataFrame(matriz_time, index=etiquetas, columns=etiquetas)
    
    return df_dist, df_time