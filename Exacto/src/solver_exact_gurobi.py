"""
Modulo 3: Solver Exacto MILP (PCVRP puro con Gurobi)
"""
import time
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Any

def resolver_pcvrp_benchmark(df_demanda: pd.DataFrame, df_dist: pd.DataFrame, df_time: pd.DataFrame, num_vehiculos: int, time_limit: int, mip_gap: float= 0) -> Dict[str, Any]:
    nodos_clientes = df_demanda['Cod. Contenedor'].tolist()
    N_0 = ['PIRS'] + nodos_clientes
    DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
    VEHICULOS = list(range(1, num_vehiculos + 1))
    
    CAPACIDAD_Q = 10000.0
    MAX_HORAS_TURNO = 7.5
    TIEMPO_SERVICIO_H = 2.5 / 60.0
    M_PENALIZACION = 500
    
    frecuencia = {row['Cod. Contenedor']: row['Opt_Frecuencia_Semanal'] for _, row in df_demanda.iterrows()}
    demanda = {row['Cod. Contenedor']: min(row['Tasa_Pesimista_Kg_Dia'] * (7.0 / row['Opt_Frecuencia_Semanal']), row['Umbral_Recogida_Efectivo']) for _, row in df_demanda.iterrows()}
    demanda['PIRS'] = 0.0
    
    dist = df_dist.to_dict('index')
    tiempo = df_time.to_dict('index')
    arcos = [(i, j) for i in N_0 for j in N_0 if i != j]

    m = gp.Model("PCVRP_Benchmark")
    m.setParam('TimeLimit', time_limit)
    m.setParam('OutputFlag', 1) # Lo activamos para que el log tenga sentido
    m.setParam('MIPGap', mip_gap)
    
    # LOG NATIVO DE GUROBI
    import os
    os.makedirs("./outputs/logs/", exist_ok=True)
    m.setParam('LogFile', f'./outputs/logs/gurobi_benchmark_{len(nodos_clientes)}_nodos.log')

    # Variables
    x = m.addVars(arcos, VEHICULOS, DIAS, vtype=GRB.BINARY, name="x")
    y = m.addVars(nodos_clientes, VEHICULOS, DIAS, vtype=GRB.BINARY, name="y")
    z = m.addVars(VEHICULOS, DIAS, vtype=GRB.BINARY, name="z")
    u = m.addVars(nodos_clientes, VEHICULOS, DIAS, vtype=GRB.CONTINUOUS, name="u")

    # Funcion Objetivo
    obj = gp.quicksum(dist[i][j] * x[i,j,k,t] for i,j in arcos for k in VEHICULOS for t in DIAS) + \
          gp.quicksum(M_PENALIZACION * z[k,t] for k in VEHICULOS for t in DIAS)
    m.setObjective(obj, GRB.MINIMIZE)

    # Restricciones
    for i in nodos_clientes:
        m.addConstr(gp.quicksum(y[i,k,t] for k in VEHICULOS for t in DIAS) >= frecuencia[i])

    for i in nodos_clientes:
        for k in VEHICULOS:
            for t in DIAS:
                m.addConstr(gp.quicksum(x[i,j,k,t] for j in N_0 if j!=i) == y[i,k,t])
                m.addConstr(gp.quicksum(x[j,i,k,t] for j in N_0 if j!=i) == y[i,k,t])
                
    for k in VEHICULOS:
        for t in DIAS:
            m.addConstr(gp.quicksum(x['PIRS',j,k,t] for j in nodos_clientes) == z[k,t])
            m.addConstr(gp.quicksum(x[j,'PIRS',k,t] for j in nodos_clientes) == z[k,t])

    for k in VEHICULOS:
        for t in DIAS:
            for i in nodos_clientes:
                m.addConstr(u[i,k,t] >= demanda[i] * y[i,k,t])
                m.addConstr(u[i,k,t] <= CAPACIDAD_Q * y[i,k,t])
                for j in nodos_clientes:
                    if i != j:
                        m.addConstr(u[i,k,t] + demanda[j] - CAPACIDAD_Q * (1 - x[i,j,k,t]) <= u[j,k,t])

    for k in VEHICULOS:
        for t in DIAS:
            t_cond = gp.quicksum(tiempo[i][j] * x[i,j,k,t] for i,j in arcos)
            t_serv = gp.quicksum(TIEMPO_SERVICIO_H * y[i,k,t] for i in nodos_clientes)
            m.addConstr(t_cond + t_serv <= MAX_HORAS_TURNO * z[k,t])

    start_time = time.time()
    m.optimize()
    runtime = time.time() - start_time

    status_str = {GRB.OPTIMAL: 'OPTIMAL', GRB.TIME_LIMIT: 'TIME_LIMIT', GRB.INFEASIBLE: 'INFEASIBLE'}.get(m.Status, str(m.Status))
    
    result_metrics = {
        'Runtime_Sec': round(runtime, 2),
        'Status': status_str,
        'ObjVal': round(m.ObjVal, 2) if m.SolCount > 0 else None,
        'MIPGap': round(m.MIPGap, 4) if hasattr(m, 'MIPGap') and m.SolCount > 0 else None
    }
    
    return result_metrics