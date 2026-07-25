'''This module answers the operations research question. Give limit resources, which batteries 
should we replace?'''
from ortools.linear_solver import pywraplp
from inference.predict import predict_soh_batteries, health_report
import numpy as np
import pandas as pd
from .plots import plot_soh_curve, plot_rul, plot_risk

# %%
def make_health_report(folder_path):
    '''make health report for the batteries
    Input: the folder having batteries
    Output: Th generated health report'''
    df = predict_soh_batteries(folder_path)
    dm = health_report(df)
    return dm, df

# %%
def health_report_with_parameters(health_report, csv_path):
    '''This function merges the predicted technical parameters of the battery with 
    the operational parameters so that OR algorithms can run. The operational parameters 
    are stored in a seperate csv file(see readme)'''
    data = pd.read_csv(csv_path)
    merged_report = health_report.merge(data, on='Battery_ID', how='left')
    return merged_report



# %%
def optimize_replacement(merged_report, budget,max_hours, max_replacements):
    '''This is the main function to optimize battery replacement. 
    Objective: Maximize the benefit from the battery replacement sunject to budget, maintainence hours,
    maximum replacements. '''
    df = merged_report
    solver = pywraplp.Solver.CreateSolver('SCIP')# Mixed Integer Linear Programming solver
    n=len(df)
    replace = [solver.BoolVar(f'replace_{i}') for i in range(n)]# binary decision variable-0 keep battery; 1 replace
    costs = df['Replacement_Cost'].tolist()
    criticality = df['Criticality'].tolist()
    rep_time = df['Replacement_Time'].tolist()
    risks = df['Risk'].tolist()
    objective = solver.Objective()
    benefits = [risks[i]*criticality[i] for i in range(n)]#Higher-risk and higher-criticality batteries provide greater benefit when replaced.
    for i in range(n):
        objective.SetCoefficient(replace[i], benefits[i])
    objective.SetMaximization()#Maximize fleet benefit
    budget_constraint = solver.Constraint(0, budget)#Total replacement cost must stay within budget.
    hour_constraint = solver.Constraint(0, max_hours)#Total maintenance hours available.
    replacement_constraint = solver.Constraint(0, max_replacements)# Maximum number of batteries that can be replaced.
    for i in range(n):
        budget_constraint.SetCoefficient(replace[i], costs[i]) 
        hour_constraint.SetCoefficient(replace[i], rep_time[i])
        replacement_constraint.SetCoefficient(replace[i], 1)
    status = solver.Solve()
    if status!=pywraplp.Solver.OPTIMAL:
        print('No optimal solution')
        return None 
    result = df.copy()
    result['Replace']=[int(replace[i].solution_value()) for i in range(n)]
    return result

    

# %%
def generate_maintainence_plan(result):
    '''Extract batteries for replacement'''
    maintainence_plan = result[result['Replace']==1][['Battery_ID', 'Replacement_Cost', 'Replacement_Time', 'Current_SOH', 'Remaining_useful_life', 'Risk', 'Health_Status']]
    return maintainence_plan

# %%
def generate_fleet_summary(result, budget, max_hours):
    '''Compute statistics for the dashboard.'''
    total_batts = len(result)
    healthy = (result['Health_Status']=='Healthy').sum()
    monitor = (result['Health_Status']=='Monitor').sum()
    warning = (result['Health_Status']=='Warning').sum()
    critical = (result['Health_Status']=='Critical').sum()
    avg_soh = result['Current_SOH'].mean()
    avg_rul = result['Remaining_useful_life'].mean()
    Fleet_risk_before = result['Risk'].sum()
    selected = result[result['Replace']==1]
    budget_used=selected['Replacement_Cost'].sum()
    hours_used = selected["Replacement_Time"].sum()
    batteries_replaced = selected.shape[0]
    Fleet_risk_after = (result["Risk"] * (1 - result["Replace"])).sum()
    risk_red = (Fleet_risk_before-Fleet_risk_after)*100/Fleet_risk_before if Fleet_risk_before != 0 else 0
    return {'Total_Batteries':total_batts,
            'Healthy': healthy,
            'Monitor':monitor,
            'Warning':warning,
            'Critical':critical,
            'Average_SOH':avg_soh,
            'Average_Remaining_Useful_Life': avg_rul,
            'Fleet_Risk_Before':Fleet_risk_before,
            'Fleet_Risk_After': Fleet_risk_after,
            'Risk_Reduced': risk_red,
            'Budget_Alloted': budget,
            'Hours_Alloted': max_hours,
            'Budget_Used': budget_used,
            'Hours_Used': hours_used,
            'Batteries_Replaced': batteries_replaced}
    
    


# %%

# %%
def run_pipeline(folder_path, csv_path, budget, max_hours, max_replacements):
    '''execute the full pipeline as one function'''
    health, prediction_df = make_health_report(folder_path)
    df = health_report_with_parameters(health, csv_path)# Merge operational parameters
    result = optimize_replacement(df, budget, max_hours, max_replacements)# Optimization
    maintainence_plan = generate_maintainence_plan(result)
    fleet_summary = generate_fleet_summary(result, budget, max_hours)#Dashboard summaries
    
    soh_plots={}#Visualizations
    for battery in prediction_df["Battery_ID"].unique():
        soh_plots[battery] = plot_soh_curve(
            prediction_df,
            battery,
        )

    rul_plot = plot_rul(health)
    risk_plot = plot_risk(health)
    return {
        "prediction": prediction_df,
        "health_report": health,
        "optimization": result,
        "maintenance_plan": maintainence_plan,
        "fleet_summary": fleet_summary,
        "plots": {
            "soh": soh_plots,
            "rul": rul_plot,
            "risk": risk_plot,
        },
    }


