# Heuristic Optimization for Dynamic Routing in Municipal Organic Waste Collection ♻️🚛

> **Master's Thesis Project (TFM)** > **Author:** Mario Pérez de Saralegui  
> **Institution:** Universidad de La Laguna (ULL)  

This repository contains the complete Python source code and data pipelines developed for solving the **Periodic Capacitated Vehicle Routing Problem (PCVRP)** applied to the municipal organic waste collection network in San Cristóbal de La Laguna. 

By transitioning from a rigid, static collection schedule to a data-driven, dynamic metaheuristic framework, this project successfully minimizes weekly travel distances, eradicates isolated fuel-wasting trips, and strictly enforces real-world operational constraints (10,000 kg vehicle payload and 7.5-hour legal shifts).

## 🧠 Project Architecture

The algorithmic framework is built on a highly modular, Object-Oriented Programming (OOP) architecture. It is divided into three main layers:

1. **Data Processing & Modeling:** - Processes raw telemetry logs to estimate robust stochastic generation rates (MAD).
   - Queries the **OSRM API** to build highly accurate real-world distance and time matrices.
2. **Exact Solvers Benchmarking:** - Mathematical formulations implemented via **Gurobi Optimizer** and **Google OR-Tools (CP-SAT)**, demonstrating the NP-hard combinatorial explosion of the real-world network (culminating in OOM limits).
3. **Metaheuristic Optimization (The Core):**
   - **Phase 1 (Construction):** Reactive GRASP (Greedy Randomized Adaptive Search Procedure) to generate initial feasible weekly schedules.
   - **Phase 2 (Local Search):** VND (Variable Neighborhood Descent) utilizing 2-opt, Or-opt, and Swap operators to polish routes.
   - **Phase 3 (Global Search):** VNS (Variable Neighborhood Search) injecting controlled stochastic noise (*shaking*) to escape local minima and converge toward pseudo-optimal structures.

## 📂 Repository Structure

```code
```exact
├── data/                   # Telemetry logs and GPS coordinates
├── outputs/                # Outputs from the main scripts
├── src/                    # Main Python source code
│   ├── data_processing/    # EDA, generation rates, and API handlers
│   └── exact_solvers/      # MILP & CP-SAT models for benchmarking
└── main.py                 # Main execution scripts

```heuristic
├── data/                   # Telemetry logs and GPS coordinates
├── outputs/                # Outputs from the main scripts
├── maps/                   # Output examples of the maps shown in figs 5.1 to 5.6 in memo
├── src/                    # Main Python source code
│   ├── data_processing/    # EDA, generation rates, and API handlers
│   └── metaheuristics/     # GRASP, VND, and VNS algorithms
└── main.py                 # Main execution scripts
