import pandas as pd
from pulp import *
import matplotlib.pyplot as plt

# Reading the data from Excel
df_submarkets = pd.read_excel("Inputs.xlsx", sheet_name="Submarkets")
inflows = pd.read_excel("Inputs.xlsx", sheet_name="Inflows")
df_thermal_plants = pd.read_excel("Inputs.xlsx", sheet_name="Thermals")
df_hydroelectric_plants = pd.read_excel("Inputs.xlsx", sheet_name="Hydroelectrics")
df_load = pd.read_excel("Inputs.xlsx", sheet_name="Load")
df_interconnections = pd.read_excel("Inputs.xlsx", sheet_name="Interchanges")

thermal_plants_submarket = []
hydroelectric_plants_submarket = []

for i, s in df_submarkets.iterrows():
    thermal_plants_submarket.append(df_thermal_plants[df_thermal_plants.submarket == s.iloc[0]].reset_index(drop=True))
    hydroelectric_plants_submarket.append(df_hydroelectric_plants[df_hydroelectric_plants.submarket == s.iloc[0]].reset_index(drop=True))

n_stages = len(df_load.T) - 1
n_submarkets = len(df_submarkets)

# Define the problem
problem = LpProblem("DispatchProblem", LpMinimize)

# Variables
thermal_plant_generation = LpVariable.dicts("thermal_plant_generation", 
                                            [(t, s, i) for s in range(n_submarkets) 
                                                 for t in range(len(thermal_plants_submarket[s])) 
                                                 for i in range(n_stages)], lowBound=0)

hydroelectric_plant_generation = LpVariable.dicts("hydroelectric_plant_generation", 
                                                  [(h, s, i) for s in range(n_submarkets) 
                                                       for h in range(len(hydroelectric_plants_submarket[s])) 
                                                       for i in range(n_stages)], lowBound=0)

reservoir_level = LpVariable.dicts("reservoir_level", 
                                  [(h, s, i) for s in range(n_submarkets) 
                                           for h in range(len(hydroelectric_plants_submarket[s])) 
                                           for i in range(n_stages)], lowBound=0)

natural_energy_inflow = LpVariable.dicts("natural_energy_inflow", 
                                        [(h, s, i) for s in range(n_submarkets) 
                                                 for h in range(len(hydroelectric_plants_submarket[s])) 
                                                 for i in range(n_stages)], lowBound=0)

spill = LpVariable.dicts("spill", 
                         [(h, s, i) for s in range(n_submarkets) 
                                  for h in range(len(hydroelectric_plants_submarket[s])) 
                                  for i in range(n_stages)], lowBound=0)

thermal_generation_submarket = LpVariable.dicts("thermal_generation_total", 
                                               [(s, i) for s in range(n_submarkets) 
                                                        for i in range(n_stages)], lowBound=0)

hydroelectric_generation_submarket = LpVariable.dicts("hydroelectric_generation_total", 
                                                      [(s, i) for s in range(n_submarkets) 
                                                               for i in range(n_stages)], lowBound=0)

import_energy = LpVariable.dicts("import_energy", 
                                [(s, i) for s in range(n_submarkets) 
                                         for i in range(n_stages)], lowBound=0)

export_energy = LpVariable.dicts("export_energy", 
                                [(s, i) for s in range(n_submarkets) 
                                         for i in range(n_stages)], lowBound=0)

interchanges = LpVariable.dicts("interchanges", 
                                [(o, d, i) for o in range(n_submarkets) 
                                           for d in range(n_submarkets) 
                                           for i in range(n_stages)], lowBound=0)

# Load Deficit Variable (new)
load_deficit = LpVariable.dicts("load_deficit", 
                                [(s, i) for s in range(n_submarkets) 
                                         for i in range(n_stages)], lowBound=0)

penalty_cost = 6000  # Cost per MW of load deficit

# Adding dual variable tracking for load balance constraints
load_balance_constraints = []

# Objective function with load deficit cost
problem += lpSum([thermal_plant_generation[t, s, i] * thermal_plants_submarket[s].cvu[t] 
                  for s in range(n_submarkets) 
                  for t in range(len(thermal_plants_submarket[s])) 
                  for i in range(n_stages)]) \
           + lpSum([hydroelectric_plant_generation[h, s, i] * 0.0  # Assuming hydro generation has no cost
                    for s in range(n_submarkets) 
                    for h in range(len(hydroelectric_plants_submarket[s])) 
                    for i in range(n_stages)]) \
           + lpSum([penalty_cost * load_deficit[s, i] 
                    for s in range(n_submarkets) 
                    for i in range(n_stages)])

# Constraints for Load Deficit
for i in range(n_stages):
    for s in range(n_submarkets):
        # Load balance equation
        constraint = thermal_generation_submarket[s, i] + hydroelectric_generation_submarket[s, i] + import_energy[s, i] - export_energy[s, i] + load_deficit[s, i] == df_load.iloc[s, i + 1]
        problem += constraint  

        load_balance_constraints.append(constraint)

        # Thermal and hydro generation aggregations
        problem += lpSum([thermal_plant_generation[t, s, i] for t in range(len(thermal_plants_submarket[s]))]) == thermal_generation_submarket[s, i]
        problem += lpSum([hydroelectric_plant_generation[h, s, i] for h in range(len(hydroelectric_plants_submarket[s]))]) == hydroelectric_generation_submarket[s, i]

        problem += lpSum([interchanges[s, d, i] for d in range(n_submarkets)]) == export_energy[s, i]
        problem += lpSum([interchanges[o, s, i] for o in range(n_submarkets)]) == import_energy[s, i]

        for t in range(len(thermal_plants_submarket[s])):
            problem += thermal_plant_generation[t, s, i] <= thermal_plants_submarket[s].capacity[t]
        
        for h in range(len(hydroelectric_plants_submarket[s])):
            problem += hydroelectric_plant_generation[h, s, i] <= hydroelectric_plants_submarket[s].capacity[h]            
            problem += reservoir_level[h, s, i] <= hydroelectric_plants_submarket[s].reservoir_capacity[h]
            problem += natural_energy_inflow[h, s, i] <= hydroelectric_plants_submarket[s].productivity[h] * hydroelectric_plants_submarket[s].capacity[h] * inflows.iloc[s, i + 1]
            
            if i > 0:
                problem += natural_energy_inflow[h, s, i] + (reservoir_level[h, s, i - 1] - reservoir_level[h, s, i]) - spill[h, s, i] == hydroelectric_plant_generation[h, s, i]
            else:
                initial_level = hydroelectric_plants_submarket[s].reservoir_level[h] * hydroelectric_plants_submarket[s].reservoir_capacity[h]
                problem += natural_energy_inflow[h, s, i] + (initial_level - reservoir_level[h, s, i]) - spill[h, s, i] == hydroelectric_plant_generation[h, s, i]

        for d in range(n_submarkets):
            problem += interchanges[d, s, i] <= df_interconnections.iloc[d, s + 1]
            problem += interchanges[s, d, i] <= df_interconnections.iloc[s, d + 1]

# Solve the optimization problem
problem.solve(PULP_CBC_CMD(msg=1))

# Results
print("Current Status =", LpStatus[problem.status])


for s in range(n_submarkets):
    for i in range(n_stages):
        print(f"Submarket {df_submarkets.iloc[s, 0]}, Stage {i + 1}:")
        print(f"Thermal Generation: {thermal_generation_submarket[s, i].varValue} MW")
        print(f"Hydro Generation: {hydroelectric_generation_submarket[s, i].varValue} MW")

        # Accessing the dual variable (marginal cost) associated with the load balance constraint
        # Note: The dual variable corresponds to the 'Load Balance' constraint
        marginal_cost = load_balance_constraints[s * n_stages + i].pi
        print(f"Submarket {df_submarkets.iloc[s, 0]}, Stage {i + 1}: Marginal Cost = {marginal_cost} $/MW")

# Display the results (load deficit for each submarket)
for s in range(n_submarkets):
    for i in range(n_stages):
        print(f"Submarket {df_submarkets.iloc[s, 0]}, Stage {i + 1}: Load Deficit = {load_deficit[s, i].varValue} MW")

# Plot interface flows between submarkets for each pair (excluding self-loops)
for s in range(n_submarkets):
    thermal = []
    hydro = []
    load = []
    for i in range(n_stages):
        thermal.append(thermal_generation_submarket[s, i].varValue)
        hydro.append(hydroelectric_generation_submarket[s, i].varValue)
        load.append(df_load.iloc[s, i + 1])

    print(f"\nSubmarket {df_submarkets.iloc[s, 0]}:")
    print("Stage | Thermal | Hydro | Load")
    for i in range(n_stages):
        print(f"{i + 1:5} | {thermal[i]:7.2f} | {hydro[i]:5.2f} | {load[i]:4.2f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_stages + 1), thermal, label='Thermal Dispatch')
    plt.plot(range(1, n_stages + 1), hydro, label='Hydro Dispatch')
    plt.plot(range(1, n_stages + 1), load, label='Load', linestyle='--')
    plt.title(f"Dispatch and Load - Submarket {df_submarkets.iloc[s, 0]}")
    plt.xlabel("Stage")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
