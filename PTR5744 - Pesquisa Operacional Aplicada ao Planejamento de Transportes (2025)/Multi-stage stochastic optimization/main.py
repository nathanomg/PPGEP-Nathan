"""
This script formulates and solves a multi-stage stochastic optimization problem for the operation planning of an electric power system with hydrothermal generation and interconnections between subsystems (submarkets).
Main Steps:
-----------
1. **Data Import**:
    - Reads input data from an Excel file (`Inputs.xlsx`) for submarkets, inflows, thermal and hydro plants, load, and interconnections.
2. **Data Organization**:
    - Groups thermal and hydro plants by submarket for easier access during model formulation.
3. **Model Parameters**:
    - Determines the number of stages (time periods) and submarkets.
4. **Decision Variables**:
    - Defines variables for generation (thermal and hydro), reservoir levels, natural energy inflow (ENA), spillage, total generation per submarket, imports/exports, and interchanges between submarkets.
5. **Objective Function**:
    - Minimizes the total operational cost, considering the variable cost of thermal generation.
6. **Constraints**:
    - **Load Balance**: Ensures supply meets demand in each submarket and stage.
    - **Generation Aggregation**: Relates individual plant generation to total submarket generation.
    - **Interchange Balance**: Relates imports/exports to interchanges between submarkets.
    - **Capacity Limits**: Enforces generation and reservoir capacity limits for each plant.
    - **Hydrological Balance**: Models water balance in reservoirs, including inflows, outflows, and spillage.
    - **Interchange Limits**: Enforces limits on energy exchanges between submarkets.
7. **Model Solution**:
    - Solves the optimization problem using the CBC solver.
8. **Results**:
    - Prints the solution status.
Notes:
------
- The script assumes the input Excel file is properly formatted and available in the working directory.
- The model can be extended to include more detailed operational constraints or stochastic elements.
"""

import pandas as pd
from pulp import *

df_submercados   = pd.read_excel("Inputs.xlsx", sheet_name="Submercados")
vazoes           = pd.read_excel("Inputs.xlsx", sheet_name="Vazões")
df_termicas      = pd.read_excel("Inputs.xlsx", sheet_name="Térmicas")
df_hidreletricas = pd.read_excel("Inputs.xlsx", sheet_name="Hidrelétricas")
df_carga         = pd.read_excel("Inputs.xlsx", sheet_name="Carga")
df_intercambios  = pd.read_excel("Inputs.xlsx", sheet_name="Intercâmbios")

termicas_submercado      = []
hidreletricas_submercado = []


for i,s in df_submercados.iterrows():
    termicas_submercado.append(df_termicas[df_termicas.submercado == s.iloc[0]].reset_index(drop=True))
    hidreletricas_submercado.append(df_hidreletricas[df_hidreletricas.submercado == s.iloc[0]].reset_index(drop=True))   

n_estagios     = len(df_carga.T)-1
n_submercados  = len(df_submercados)

problem = LpProblem("ProblemaDespacho", LpMinimize)

ger_termica_usina  = LpVariable.dicts("ger_termica_usina",  [(t,s,i) for s in range(n_submercados) for t in range(len(termicas_submercado[s]))      for i in range(n_estagios)], lowBound=0)
ger_hidr_usina     = LpVariable.dicts("ger_hidr_usina",     [(h,s,i) for s in range(n_submercados) for h in range(len(hidreletricas_submercado[s])) for i in range(n_estagios)], lowBound=0)
reservatorio_usina = LpVariable.dicts("reservatorio_usina", [(h,s,i) for s in range(n_submercados) for h in range(len(hidreletricas_submercado[s])) for i in range(n_estagios)], lowBound=0)
ena_usina          = LpVariable.dicts("ena_usina",          [(h,s,i) for s in range(n_submercados) for h in range(len(hidreletricas_submercado[s])) for i in range(n_estagios)], lowBound=0)
vertimento_usina   = LpVariable.dicts("vertimento_usina",   [(h,s,i) for s in range(n_submercados) for h in range(len(hidreletricas_submercado[s])) for i in range(n_estagios)], lowBound=0)
ger_termica_subm   = LpVariable.dicts("ger_termica_total",  [(s,i)   for s in range(n_submercados) for i in range(n_estagios)], lowBound=0)
ger_hidr_subm      = LpVariable.dicts("ger_hidr_total",     [(s,i)   for s in range(n_submercados) for i in range(n_estagios)], lowBound=0)
importacao         = LpVariable.dicts("importacao",         [(s,i)   for s in range(n_submercados) for i in range(n_estagios)], lowBound=0)
exportacao         = LpVariable.dicts("exportacao",         [(s,i)   for s in range(n_submercados) for i in range(n_estagios)], lowBound=0)
intercambios       = LpVariable.dicts("intercambios",       [(o,d,i) for o in range(n_submercados) for d in range(n_submercados) for i in range(n_estagios)], lowBound=0)

#Objective function
problem += lpSum([ger_termica_usina[t,s,i]*termicas_submercado[s].cvu[t] for s in range(n_submercados) for t in range(len(termicas_submercado[s])) for i in range(n_estagios)])

#Constraints
for i in range(n_estagios):
    for s in range(n_submercados):
        problem += ger_hidr_subm[s,i] + ger_termica_subm[s,i] + importacao[s,i] - exportacao[s,i] == df_carga.iloc[s,i+1]
        
        problem += lpSum([ger_termica_usina[t,s,i] for t in range(len(termicas_submercado[s]))]) == ger_termica_subm[s,i]
        problem += lpSum([ger_hidr_usina[h,s,i]    for h in range(len(hidreletricas_submercado[s]))]) == ger_hidr_subm[s,i]
        
        problem += lpSum([intercambios[s,d,i] for d in range(n_submercados)]) == exportacao[s,i]
        problem += lpSum([intercambios[o,s,i] for o in range(n_submercados)]) == importacao[s,i]

        for t in range(len(termicas_submercado[s])):        problem += ger_termica_usina[t,s,i]  <= termicas_submercado[s].capacidade[t]
        for h in range(len(hidreletricas_submercado[s])): 
            problem += ger_hidr_usina[h,s,i]     <= hidreletricas_submercado[s].capacidade[h]            
            
            problem += reservatorio_usina[h,s,i] <= hidreletricas_submercado[s].capacidade_reservatorio[h]

            problem += ena_usina[h,s,i] <= hidreletricas_submercado[s].produtibilidade[h] * hidreletricas_submercado[s].capacidade[h]*vazoes.iloc[s,i+1]
            
            if i > 0:
                problem += ena_usina[h,s,i] + (reservatorio_usina[h,s,i-1] - reservatorio_usina[h,s,i]) - vertimento_usina[h,s,i]  ==  ger_hidr_usina[h,s,i]
            else:
                nivel_inicial = hidreletricas_submercado[s].nivel_reservatorio[h] * hidreletricas_submercado[s].capacidade_reservatorio[h]
                problem += ena_usina[h,s,i] + (nivel_inicial - reservatorio_usina[h,s,i]) - vertimento_usina[h,s,i]  ==  ger_hidr_usina[h,s,i]

        for d in range(n_submercados):
            problem += intercambios[d,s,i] <= df_intercambios.iloc[d,s+1]
            problem += intercambios[s,d,i] <= df_intercambios.iloc[s,d+1]
    

#Solve
problem.solve(PULP_CBC_CMD(msg=1))

# #Resultados
# for v in problem.variables():
#     print(v.name, "=", v.varValue)

# print("FO =", value(problem.objective))


print("Current Status =", LpStatus[problem.status])

# Plot interface flows between submarkets for each pair (excluding self-loops)

import matplotlib.pyplot as plt

# Prepare results for each submarket

for s in range(n_submercados):
    thermal = []
    hydro = []
    load = []
    for i in range(n_estagios):
        thermal.append(ger_termica_subm[s, i].varValue)
        hydro.append(ger_hidr_subm[s, i].varValue)
        load.append(df_carga.iloc[s, i+1])

    print(f"\nSubmarket {df_submercados.iloc[s,0]}:")
    print("Stage | Thermal | Hydro | Load")
    for i in range(n_estagios):
        print(f"{i+1:5} | {thermal[i]:7.2f} | {hydro[i]:5.2f} | {load[i]:4.2f}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, n_estagios+1), thermal, label='Thermal Dispatch')
    plt.plot(range(1, n_estagios+1), hydro, label='Hydro Dispatch')
    plt.plot(range(1, n_estagios+1), load, label='Load', linestyle='--')
    plt.title(f"Dispatch and Load - Submarket {df_submercados.iloc[s,0]}")
    plt.xlabel("Stage")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for o in range(n_submercados):
    for d in range(n_submercados):
        if o != d:
            flows = [intercambios[o, d, i].varValue for i in range(n_estagios)]
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, n_estagios+1), flows, marker='o')
            plt.title(f"Interface Flow: {df_submercados.iloc[o,0]} → {df_submercados.iloc[d,0]}")
            plt.xlabel("Stage")
            plt.ylabel("Flow (MW)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()