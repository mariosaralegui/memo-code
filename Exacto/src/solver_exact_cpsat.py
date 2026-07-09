"""
Modulo 3b: Solver Exacto MILP con Google CP-SAT
"""
import time
import pandas as pd
from ortools.sat.python import cp_model
from typing import Dict, Any, Tuple

def resolver_pcvrp_cpsat(df_demanda: pd.DataFrame, df_dist: pd.DataFrame, df_time: pd.DataFrame, num_vehiculos: int, time_limit: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    
    # Escalas (CP-SAT exige enteros)
    SCALE_DIST = 1000
    SCALE_TIME = 3600
    SCALE_DEMAND = 10
    
    nodos_clientes = df_demanda['Cod. Contenedor'].tolist()
    N_0 = ['PIRS'] + nodos_clientes
    DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
    VEHICULOS = list(range(1, num_vehiculos + 1))
    
    CAPACIDAD_Q = int(10000.0 * SCALE_DEMAND)
    MAX_SEGUNDOS_TURNO = int(7.5 * SCALE_TIME)
    TIEMPO_SERVICIO_S = int((2.5 / 60.0) * SCALE_TIME)
    M_PENALIZACION = 500000 
    
    frecuencia = {row['Cod. Contenedor']: int(row['Opt_Frecuencia_Semanal']) for _, row in df_demanda.iterrows()}
    demanda = {row['Cod. Contenedor']: int(min(row['Tasa_Pesimista_Kg_Dia'] * (7.0 / row['Opt_Frecuencia_Semanal']), row['Umbral_Recogida_Efectivo']) * SCALE_DEMAND) for _, row in df_demanda.iterrows()}
    demanda['PIRS'] = 0

    model = cp_model.CpModel()
    
    # Variables
    x = {}
    for i in N_0:
        for j in N_0:
            if i != j:
                for k in VEHICULOS:
                    for t in DIAS:
                        x[i, j, k, t] = model.NewBoolVar(f'x_{i}_{j}_{k}_{t}')
                        
    y = {}
    for i in nodos_clientes:
        for k in VEHICULOS:
            for t in DIAS:
                y[i, k, t] = model.NewBoolVar(f'y_{i}_{k}_{t}')
                
    z = {}
    for k in VEHICULOS:
        for t in DIAS:
            z[k, t] = model.NewBoolVar(f'z_{k}_{t}')

    # Variable U (Carga Acumulada) para prevencion de subtours logicos
    u = {}
    for i in nodos_clientes:
        for k in VEHICULOS:
            for t in DIAS:
                u[i, k, t] = model.NewIntVar(0, CAPACIDAD_Q, f'u_{i}_{k}_{t}')

    # Restricciones Basicas
    for i in nodos_clientes:
        model.Add(sum(y[i, k, t] for k in VEHICULOS for t in DIAS) >= frecuencia[i])

    for i in nodos_clientes:
        for k in VEHICULOS:
            for t in DIAS:
                model.Add(sum(x[i, j, k, t] for j in N_0 if j != i) == y[i, k, t])
                model.Add(sum(x[j, i, k, t] for j in N_0 if j != i) == y[i, k, t])

    for k in VEHICULOS:
        for t in DIAS:
            model.Add(sum(x['PIRS', j, k, t] for j in nodos_clientes) == z[k, t])
            model.Add(sum(x[j, 'PIRS', k, t] for j in nodos_clientes) == z[k, t])
            
            model.Add(sum(demanda[i] * y[i, k, t] for i in nodos_clientes) <= CAPACIDAD_Q)

            tiempo_conduccion = sum(int(df_time.loc[i, j] * SCALE_TIME) * x[i, j, k, t] for i in N_0 for j in N_0 if i != j)
            tiempo_servicio = sum(TIEMPO_SERVICIO_S * y[i, k, t] for i in nodos_clientes)
            model.Add(tiempo_conduccion + tiempo_servicio <= MAX_SEGUNDOS_TURNO * z[k, t])

    # Eliminacion de Subtours Estricta
    for k in VEHICULOS:
        for t in DIAS:
            for i in nodos_clientes:
                model.Add(u[i, k, t] == 0).OnlyEnforceIf(y[i, k, t].Not())
                model.Add(u[i, k, t] >= demanda[i]).OnlyEnforceIf(y[i, k, t])
                
                model.Add(u[i, k, t] == demanda[i]).OnlyEnforceIf(x['PIRS', i, k, t])

                for j in nodos_clientes:
                    if i != j:
                        model.Add(u[j, k, t] == u[i, k, t] + demanda[j]).OnlyEnforceIf(x[i, j, k, t])

    # Funcion Objetivo
    objetivo_distancia = sum(int(df_dist.loc[i, j] * SCALE_DIST) * x[i, j, k, t] for i in N_0 for j in N_0 if i != j for k in VEHICULOS for t in DIAS)
    objetivo_flota = sum(M_PENALIZACION * z[k, t] for k in VEHICULOS for t in DIAS)
    
    model.Minimize(objetivo_distancia + objetivo_flota)

    # Resolucion
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.relative_gap_limit = 0.01
    solver.parameters.log_search_progress = True

    
    start_time = time.time()
    status = solver.Solve(model)
    runtime = time.time() - start_time

    status_map = {cp_model.OPTIMAL: 'OPTIMAL', cp_model.FEASIBLE: 'FEASIBLE', cp_model.INFEASIBLE: 'INFEASIBLE', cp_model.UNKNOWN: 'UNKNOWN'}
    gap = 0.0 if status == cp_model.OPTIMAL else (solver.ObjectiveValue() - solver.BestObjectiveBound()) / solver.ObjectiveValue() if solver.ObjectiveValue() > 0 else None

    metricas = {
        'Runtime_Sec': round(runtime, 2),
        'Status': status_map.get(status, str(status)),
        'ObjVal': round(solver.ObjectiveValue() / SCALE_DIST, 2) if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
        'MIPGap': round(gap, 4) if gap is not None else None
    }
    
    return metricas, pd.DataFrame() # DataFrame vacio para el benchmark