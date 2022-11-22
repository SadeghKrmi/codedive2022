import pyomo.opt as po
import pyomo.environ as pe

I   = ['BTS01', 'BTS03', 'BTS04', 'BTS05', 'BTS08', 'BTS10']
M   = ['Team01', 'Team02', 'Team03']
Ci  = {'BTS01': 109, 'BTS03': 89, 'BTS04': 23, 'BTS05': 54, 'BTS08': 86, 'BTS10': 68}
Pi  = {'BTS01': 17, 'BTS03': 2, 'BTS04': 100, 'BTS05': 21, 'BTS08': 71, 'BTS10': 34}
Tm  = {'Team01': 2, 'Team02': 2, 'Team03': 3}

Aim = {
    ('BTS01', 'Team01'): 1, ('BTS01', 'Team02'): 3, ('BTS01', 'Team03'): 1.5, 
    ('BTS03', 'Team01'): 2, ('BTS03', 'Team02'): 5, ('BTS03', 'Team03'): 0.5, 
    ('BTS04', 'Team01'): 3, ('BTS04', 'Team02'): 4, ('BTS04', 'Team03'): 4, 
    ('BTS05', 'Team01'): 1, ('BTS05', 'Team02'): 1, ('BTS05', 'Team03'): 5, 
    ('BTS08', 'Team01'): 3, ('BTS08', 'Team02'): 9, ('BTS08', 'Team03'): 3, 
    ('BTS10', 'Team01'): 4, ('BTS10', 'Team02'): 5, ('BTS10', 'Team03'): 1
}

# set solver and create model
solver = po.SolverFactory('glpk')
model = pe.ConcreteModel()

# Set definition
model.i = pe.Set(initialize=I)
model.m = pe.Set(initialize=M)

# Variable declaration
model.yim = pe.Var(model.i, model.m, domain=pe.Binary)
model.wi = pe.Var(model.i, domain=pe.Binary)

# Parameter Declaration
model.ci = pe.Param(model.i, initialize=Ci)
model.pi = pe.Param(model.i, initialize=Pi)
model.tm = pe.Param(model.m, initialize=Tm)
model.aim = pe.Param(model.i, model.m, initialize=Aim)

# cost function
def costfunc(model):
    alphaC = 0.7
    alphaP = 0.3
    fC = sum(model.ci[i] * model.wi[i] for i in model.i)
    fP = sum(model.pi[i] * model.wi[i] for i in model.i)
    return alphaC * fC + alphaP * fP

model.obj = pe.Objective(rule = costfunc, sense = pe.maximize)

# contstraint definition
def ruleC1(model, i):
    return sum(model.yim[i,m] for m in model.m) == model.wi[i]

model.C1 = pe.Constraint(model.i, rule=ruleC1)

def ruleC2(model, m):
    return sum(model.aim[i,m] * model.yim[i,m] for i in model.i) <= model.tm[m]

model.C2 = pe.Constraint(model.m, rule=ruleC2)


# solve the model
solver.solve(model)

# print outout of the model
model.pprint()