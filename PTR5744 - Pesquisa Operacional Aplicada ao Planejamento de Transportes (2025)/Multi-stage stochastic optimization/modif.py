import pandas as pd
from pulp import *
import matplotlib.pyplot as plt

file_name = "Inputs.xlsx"
# file_name = "Inputs - Copia.xlsx"

# Reading the data from Excel
df_submarkets = pd.read_excel(file_name, sheet_name="Submarkets")
inflows = pd.read_excel(file_name, sheet_name="Inflows")
df_thermal_plants = pd.read_excel(file_name, sheet_name="Thermals")
df_hydroelectric_plants = pd.read_excel(file_name, sheet_name="Hydroelectrics")
df_load = pd.read_excel(file_name, sheet_name="Load")
df_interconnections = pd.read_excel(file_name, sheet_name="Interchanges")

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

# Future Cost Estimation Dictionary (using dual variables for future cost)
future_costs = {}  # This will store future costs for each submarket and stage

# Initialize the future cost values (can start with zeros or an initial guess)
for s in range(n_submarkets):
    for i in range(n_stages):
        future_costs[(s, i)] = 0  # Initial guess (can be zero)

# Function to solve the optimization problem and return the upper bound (UB) and lower bound (LB)
def solve_problem():
    problem = LpProblem("DispatchProblem", LpMinimize)

    # Objective function with load deficit cost and future cost estimation
    problem += lpSum([thermal_plant_generation[t, s, i] * thermal_plants_submarket[s].cvu[t] 
                      for s in range(n_submarkets) 
                      for t in range(len(thermal_plants_submarket[s])) 
                      for i in range(n_stages)]) \
               + lpSum([hydroelectric_plant_generation[h, s, i] * future_costs[(s, i)]  # Assuming hydro generation has no cost
                        for s in range(n_submarkets) 
                        for h in range(len(hydroelectric_plants_submarket[s])) 
                        for i in range(n_stages)]) \
               + lpSum([penalty_cost * load_deficit[s, i] 
                        for s in range(n_submarkets) 
                        for i in range(n_stages)]) 

    # Constraints for Load Deficit and Future Cost Estimation
    load_balance_constraints = {}  # Store constraints as a dictionary
    for i in range(n_stages):
        for s in range(n_submarkets):
            # Load balance equation (defining the load balance constraints)
            constraint = thermal_generation_submarket[s, i] + hydroelectric_generation_submarket[s, i] + import_energy[s, i] - export_energy[s, i] + load_deficit[s, i] == df_load.iloc[s, i + 1]
            problem += constraint  # Add the constraint to the problem
            load_balance_constraints[(s, i)] = constraint  # Save the constraint with a tuple (submarket, stage)

            # Thermal and hydro generation aggregations
            problem += lpSum([thermal_plant_generation[t, s, i] for t in range(len(thermal_plants_submarket[s]))]) == thermal_generation_submarket[s, i]
            problem += lpSum([hydroelectric_plant_generation[h, s, i] for h in range(len(hydroelectric_plants_submarket[s]))]) == hydroelectric_generation_submarket[s, i]

            problem += lpSum([interchanges[s, d, i] for d in range(n_submarkets)]) == export_energy[s, i]
            problem += lpSum([interchanges[o, s, i] for o in range(n_submarkets)]) == import_energy[s, i]

            # Add constraints for thermal plants and hydro plants capacities
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

            # Interchange limits
            for d in range(n_submarkets):
                problem += interchanges[d, s, i] <= df_interconnections.iloc[d, s + 1]
                problem += interchanges[s, d, i] <= df_interconnections.iloc[s, d + 1]

    # Solve the optimization problem
    problem.solve(PULP_CBC_CMD(msg=1))

    return load_balance_constraints

# Iteration logic
iterations = 100  # Set the maximum number of iterations
convergence_tolerance = 0.01  # Tolerance for convergence

previous_UB = float('inf')
previous_LB = float('inf')
gap = float('inf')

# List to track UB, LB, and gap for plotting
UB_list = []
LB_list = []
gap_list = []

UB = 0

for iteration in range(iterations):
    print(f"Iteration {iteration + 1}")

    # Solve the problem and get the upper bound (UB), lower bound (LB), and dual variables
    load_balance_constraints = solve_problem()

    # Calculate the average of the marginal price of the current and the next stage for each stage
    future_costs = {}


    # Calculate average marginal cost for each submarket
    avg_marginal_cost_by_submarket = {}
    for s in range(n_submarkets):
        marginal_costs = [load_balance_constraints[(s, i)].pi for i in range(n_stages)]
        avg_mc = sum(marginal_costs) / len(marginal_costs) if marginal_costs else 0
        avg_marginal_cost_by_submarket[s] = avg_mc

    # Set future_costs for each (s, i) to the average marginal cost of submarket s
    for s in range(n_submarkets):
        for i in range(n_stages):
            future_costs[(s, i)] = avg_marginal_cost_by_submarket[s]

    # Example: print or use avg_marginal_cost_by_stage as needed
    #print(avg_marginal_cost_by_stage)

    # Calculate the average future cost by submarket
    avg_future_cost_by_submarket = {}
    for s in range(n_submarkets):
        submarket_costs = [future_costs[(s, i)] for i in range(n_stages)]
        avg_future_cost_by_submarket[s] = sum(submarket_costs) / len(submarket_costs) if submarket_costs else 0

    # For UB, use the overall average of submarket averages
    avg_future_cost = sum(avg_future_cost_by_submarket.values()) / n_submarkets if n_submarkets else 0
    
    LB = UB
    UB = avg_future_cost
    # Check convergence based on the gap between UB and LB
    gap = abs(UB - LB)
    print(f"UB: {UB}, LB: {LB}, Gap: {gap}")

    # Store UB, LB, and gap for plotting
    UB_list.append(UB)
    LB_list.append(LB)
    gap_list.append(gap)

    if gap < convergence_tolerance:
        print("Convergence reached!")
        break


    print(f"End of iteration {iteration + 1}\n")

# Plot UB, LB, and Gap
plt.figure(figsize=(10, 6))
plt.plot(UB_list, label="Upper Bound (UB)")
plt.plot(LB_list, label="Lower Bound (LB)")
plt.plot(gap_list, label="Gap (UB - LB)", linestyle="--")
plt.xlabel("Iterations")
plt.ylabel("Value ($)")
plt.title("Convergence of Upper and Lower Bounds")
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.colors as mcolors

stage_names = [f"Stage {i+1}" for i in range(n_stages)]

# Marginal Cost by Stage and Submarket (Bar Plot)
marginal_cost_matrix = np.zeros((n_submarkets, n_stages))
for s in range(n_submarkets):
    for i in range(n_stages):
        marginal_cost_matrix[s, i] = load_balance_constraints[(s, i)].pi

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.8 / n_submarkets
indices = np.arange(n_stages)
for s in range(n_submarkets):
    ax.bar(indices + s * bar_width, marginal_cost_matrix[s], bar_width, label=f"{df_submarkets.iloc[s,0]}")
ax.set_xlabel("Stage")
ax.set_ylabel("Marginal Cost")
ax.set_title("Marginal Cost by Stage and Submarket")
ax.set_xticks(indices + bar_width * (n_submarkets - 1) / 2)
ax.set_xticklabels(stage_names)
ax.legend(title="Submarket")
plt.tight_layout()
plt.show()

# Sum of Reservoir Levels by Submarket and Stage (Line Plot)
reservoir_sum_matrix = np.zeros((n_submarkets, n_stages))
for s in range(n_submarkets):
    for i in range(n_stages):
        reservoir_sum = 0
        for h in range(len(hydroelectric_plants_submarket[s])):
            val = reservoir_level[h, s, i].varValue
            reservoir_sum += val if val is not None else 0
        reservoir_sum_matrix[s, i] = reservoir_sum

fig, ax = plt.subplots(figsize=(12, 6))
for s in range(n_submarkets):
    ax.plot(range(1, n_stages + 1), reservoir_sum_matrix[s], marker='o', label=f"{df_submarkets.iloc[s,0]}")
ax.set_xlabel("Stage")
ax.set_ylabel("Sum of Reservoir Levels")
ax.set_title("Sum of Reservoirs by Submarket and Stage")
ax.set_xticks(range(1, n_stages + 1))
ax.set_xticklabels(stage_names)
ax.legend(title="Submarket")
ax.grid(True)
plt.tight_layout()
plt.show()

# Flows Between Submarkets by Stage (Line Plot)
flow_matrix = np.zeros((n_submarkets, n_submarkets, n_stages))
for i in range(n_stages):
    for o in range(n_submarkets):
        for d in range(n_submarkets):
            val = interchanges[o, d, i].varValue
            flow_matrix[o, d, i] = val if val is not None else 0

plt.figure(figsize=(12, 7))
colors = list(mcolors.TABLEAU_COLORS.values())
for o in range(n_submarkets):
    for d in range(n_submarkets):
        if o != d:
            flows = flow_matrix[o, d, :]
            label = f"{df_submarkets.iloc[o,0]} → {df_submarkets.iloc[d,0]}"
            plt.plot(range(1, n_stages + 1), flows, label=label, color=colors[(o * n_submarkets + d) % len(colors)])
plt.xlabel("Stage")
plt.ylabel("Flow (MW)")
plt.title("Flows Between Submarkets by Stage")
plt.xticks(range(1, n_stages + 1), stage_names)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Flow")
plt.grid(True)
plt.tight_layout()
plt.show()

# Thermal Dispatch by Submarket and Stage (Line Plot)
thermal_dispatch_matrix = np.zeros((n_submarkets, n_stages))
for s in range(n_submarkets):
    for i in range(n_stages):
        dispatch_sum = 0
        for t in range(len(thermal_plants_submarket[s])):
            val = thermal_plant_generation[t, s, i].varValue
            dispatch_sum += val if val is not None else 0
        thermal_dispatch_matrix[s, i] = dispatch_sum

fig, ax = plt.subplots(figsize=(12, 6))
for s in range(n_submarkets):
    ax.plot(range(1, n_stages + 1), thermal_dispatch_matrix[s], marker='o', label=f"{df_submarkets.iloc[s,0]}")
ax.set_xlabel("Stage")
ax.set_ylabel("Thermal Dispatch (MW)")
ax.set_title("Thermal Dispatch by Submarket and Stage")
ax.set_xticks(range(1, n_stages + 1))
ax.set_xticklabels(stage_names)
ax.legend(title="Submarket")
ax.grid(True)
plt.tight_layout()
plt.show()
