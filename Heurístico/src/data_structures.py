"""
Estructuras de Datos Orientadas a Objetos (PCVRP)
"""
import copy
import pandas as pd
from typing import List, Dict, Set, Tuple

# Constantes Operativas
CAPACIDAD_MAX_KG = 10000.0
TURNO_MAX_HORAS = 7.5
TIEMPO_SERVICIO_H = 2.5 / 60.0

# Diccionario de Patrones de Visita permitidos
PATRONES_VISITA = {
    1: [('Lunes',), ('Martes',), ('Miércoles',), ('Jueves',), ('Viernes',)],
    2: [('Lunes', 'Miércoles'), ('Lunes', 'Jueves'), ('Martes', 'Jueves'), ('Martes', 'Viernes'), ('Miércoles', 'Viernes')],
    3: [('Lunes', 'Miércoles', 'Viernes')]
}

class Ruta:
    """Representa el viaje de UN vehículo en UN día específico."""
    def __init__(self, dia: str, id_vehiculo: int, df_dist: pd.DataFrame, df_time: pd.DataFrame):
        self.dia = dia
        self.id_vehiculo = id_vehiculo
        self.df_dist = df_dist
        self.df_time = df_time
        
        self.nodos: List[str] = ['PIRS', 'PIRS']
        self.carga_actual: float = 0.0
        self.distancia_km: float = 0.0
        self.tiempo_h: float = 0.0

    def simular_insercion(self, nodo_id: str, demanda_kg: float, posicion: int) -> Tuple[bool, float, float]:
        """
        Comprueba matemáticamente si un nodo cabe en la ruta en una posición específica.
        Retorna: (Es_Factible, Delta_Distancia, Delta_Tiempo)
        """
        if self.carga_actual + demanda_kg > CAPACIDAD_MAX_KG:
            return False, 0.0, 0.0
            
        prev_node = self.nodos[posicion - 1]
        next_node = self.nodos[posicion]
        
        delta_dist = (self.df_dist.loc[prev_node, nodo_id] + 
                      self.df_dist.loc[nodo_id, next_node] - 
                      self.df_dist.loc[prev_node, next_node])
                      
        delta_time = (self.df_time.loc[prev_node, nodo_id] + 
                      self.df_time.loc[nodo_id, next_node] - 
                      self.df_time.loc[prev_node, next_node]) + TIEMPO_SERVICIO_H
                      
        if self.tiempo_h + delta_time > TURNO_MAX_HORAS:
            return False, 0.0, 0.0
            
        return True, delta_dist, delta_time

    def aplicar_insercion(self, nodo_id: str, demanda_kg: float, posicion: int, delta_dist: float, delta_time: float):
        """Ejecuta la inserción real en memoria una vez confirmada la factibilidad."""
        self.nodos.insert(posicion, nodo_id)
        self.carga_actual += demanda_kg
        self.distancia_km += delta_dist
        self.tiempo_h += delta_time
    
    def recalcular_metricas(self):
        """
        Recalcula la distancia y el tiempo total de la ruta desde cero tras una mutación.
        (El peso total no cambia porque los operadores VND solo reordenan los nodos existentes).
        """
        distancia = 0.0
        tiempo_cond = 0.0
        
        # Sumar distancias y tiempos del nuevo orden de los nodos
        for i in range(len(self.nodos) - 1):
            n1 = self.nodos[i]
            n2 = self.nodos[i + 1]
            distancia += self.df_dist.loc[n1, n2]
            tiempo_cond += self.df_time.loc[n1, n2]
            
        # El tiempo de servicio depende solo de los contenedores reales (excluyendo los 2 PIRS)
        contenedores_reales = len(self.nodos) - 2
        tiempo_serv = contenedores_reales * TIEMPO_SERVICIO_H
        
        # Actualizar los atributos de la ruta
        self.distancia_km = distancia
        self.tiempo_h = tiempo_cond + tiempo_serv


class SolucionGlobal:
    """Contiene toda la planificación semanal (Múltiples Rutas distribuidas en 5 días)."""
    def __init__(self, df_dist: pd.DataFrame, df_time: pd.DataFrame, demandas_diarias: Dict[str, float], num_vehiculos: int = 6):
        self.df_dist = df_dist
        self.df_time = df_time
        self.demandas_diarias = demandas_diarias
        self.dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
        self.flota_max = num_vehiculos
        
        # Estructura: self.rutas['Lunes'][0] -> Ruta del camión 1 el Lunes
        self.rutas: Dict[str, List[Ruta]] = {
            dia: [Ruta(dia, k, df_dist, df_time) for k in range(1, num_vehiculos + 1)] 
            for dia in self.dias
        }
        
        self.nodos_no_asignados: Set[str] = set(demandas_diarias.keys())
        self.coste_total_km: float = 0.0

    def calcular_coste_objetivo(self) -> float:
        """
        Función Objetivo de la heurística. 
        Penaliza severamente los nodos que se queden sin recoger.
        """
        distancia_total = sum(ruta.distancia_km for dia in self.dias for ruta in self.rutas[dia])
        penalizacion_no_asignados = len(self.nodos_no_asignados) * 100000.0  # Big-M
        
        self.coste_total_km = distancia_total + penalizacion_no_asignados
        return self.coste_total_km