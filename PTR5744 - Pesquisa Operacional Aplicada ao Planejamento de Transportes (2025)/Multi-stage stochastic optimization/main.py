import pandas as pd
from pulp import *

submercados   = pd.read_excel("Inputs.xlsx", sheet_name="Submercados")
#vazoes        = pd.read_excel("Inputs.xlsx", sheet_name="Submercados")
termicas      = pd.read_excel("Inputs.xlsx", sheet_name="Térmicas")
hidreletricas = pd.read_excel("Inputs.xlsx", sheet_name="Hidrelétricas")

carga         = pd.read_excel("Inputs.xlsx", sheet_name="Carga")
carga         = carga.melt(id_vars=["submercado"], var_name="estagio", value_name="carga")

intercambios  = pd.read_excel("Inputs.xlsx", sheet_name="Intercâmbios")
intercambios  = intercambios.melt(id_vars=["origem"], var_name="destino", value_name="limite")


n_estagios     = carga.estagio.max()
n_submercados  = len(carga.submercado.unique())
n_termicas     = len(termicas)
n_hidro        = len(hidreletricas)
n_intercambios = len(intercambios)



problem = LpProblem("ProblemaDespacho", LpMinimize)

ger_termica      = LpVariable.dicts("ger_termica",  [(t,i) for t in range(n_termicas)     for i in range(n_estagios)], lowBound=0)
ger_hidr         = LpVariable.dicts("ger_hidr",     [(h,i) for h in range(n_hidro)        for i in range(n_estagios)], lowBound=0)
reservatorio     = LpVariable.dicts("reservatorio", [(h,i) for h in range(n_hidro)        for i in range(n_estagios)], lowBound=0)
intercambios     = LpVariable.dicts("intercambios", [(h,i) for h in range(n_intercambios) for i in range(n_estagios)], lowBound=0)

#Objective function
problem += lpSum([ger_termica[t,i]*termicas.cvu[t] for t in range(n_termicas) for i in range(n_estagios)])


#Constraints
for i in range(n_estagios):
    for t in range(n_termicas): problem += ger_termica[t,i]  <= termicas.capacidade[t]

    
    for s in range(n_submercados):
        problem += lpSum([ger_termica[t,i] for t in range(n_termicas)])  == combustivel_producao[c]


for i in range(n_estagios):  problem +=  lpSum([mistura_producao[c,m] for m in misturas]) == combustivel_producao[c]
for i in range(1, N): problem += estoque_A[i] == producao_A[i-1] + estoque_A[i-1] - demanda_A[i-1]

#Solve
problem.solve(PULP_CBC_CMD(msg=0))

#Resultados
for v in problem.variables():
    print(v.name, "=", v.varValue)

print("FO =", value(problem.objective))

print("Current Status =", LpStatus[problem.status])