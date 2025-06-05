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
               + lpSum([hydroelectric_plant_generation[h, s, i] * 0  # Assuming hydro generation has no cost
                        for s in range(n_submarkets) 
                        for h in range(len(hydroelectric_plants_submarket[s])) 
                        for i in range(n_stages)]) \
               + lpSum([penalty_cost * load_deficit[s, i] 
                        for s in range(n_submarkets) 
                        for i in range(n_stages)]) \
               + lpSum(future_costs[(s,i)] for s in range(n_sub) for i in range(n_stages))

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

    # Calculate upper bound (UB)
    UB = value(problem.objective)

    # Calculate lower bound (LB) from future cost estimates
    true_cost = (
        sum(thermal_plant_generation[t,s,i].value() *  thermal_plants_submarket[s].cvu[t]
            for s in range(n_submarkets) for t in range(len(thermal_plants_submarket[s]))
            for i in range(n_stages))
        + penalty_cost * sum(load_deficit[s,i].value()
            for s in range(n_submarkets) for i in range(n_stages))
    )
    UB = true_cost
    return UB, LB, load_balance_constraints

# Iteration logic
iterations = 10  # Set the maximum number of iterations
convergence_tolerance = 0.01  # Tolerance for convergence

previous_UB = float('inf')
previous_LB = float('inf')
gap = float('inf')

# List to track UB, LB, and gap for plotting
UB_list = []
LB_list = []
gap_list = []

for iteration in range(iterations):
    print(f"Iteration {iteration + 1}")

    # Solve the problem and get the upper bound (UB), lower bound (LB), and dual variables
    UB, LB, load_balance_constraints = solve_problem()

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

    # Update future costs with the dual variables (marginal costs)
    for s in range(n_submarkets):
        for i in range(n_stages):
            pi = load_balance_constraints[(s,i)].pi                 # shadow price (R$/MW)
            exp_load_next = df_load.iloc[s, i+1] if i+1 < n_stages else 0
            future_costs[(s,i)] = pi * exp_load_next                # R$ expected

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
