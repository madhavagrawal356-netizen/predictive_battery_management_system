'''This module loads the trained SOH prediction model, performs
inference on battery cycles, estimates Remaining Useful Life (RUL),
calculates risk scores, and generates battery health reports.'''
from model.model import MainModel
import pandas as pd
from data.dataset import final_dataset
from data.dataloaders import InferenceDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import r2_score

# %%
def predict(data, model):
    '''Generate SOH predictios for every battery cycle. 
    Input: Data from the dataloader and the trained model
     Output: Dataframe with predicted soh for every battery cycle '''
    rows=[]
    model.eval()
    with torch.no_grad():
        for x, battery_id, cycle_number in data:
            ypred=model(x).squeeze(1)
            for i in range(len(x)):
                rows.append({
                    "Battery_ID": battery_id[i],
                    "Cycle_Number": cycle_number[i].item(),
                    "Predicted_SOH": ypred[i].item(),
                })
    df = pd.DataFrame(rows)
    return df


# %%
def predict_soh_batteries(folder_path):
    '''Predict SOH for every battery contained in the folder
    Input: the folder path
    Output: Prediction Dataframe'''
    samples = final_dataset(folder_path, training=False)# Build inference dataset
    dat_set= DataLoader(InferenceDataset(samples), batch_size=1, shuffle=False)# Create dataloader
    model= MainModel(6)
    model.load_state_dict(torch.load(r'model\best_model.pth'))# Load trained model
    df = predict(dat_set, model)
    return df



# %%


# %%
def get_battery_history(df, battery_id):
    '''Extract the predicted SOH history of one battery.
    Output:cycle_numbers, predicted_soh'''
    battery = df[df["Battery_ID"] == battery_id]
    battery = battery.sort_values("Cycle_Number")

    cycle = battery["Cycle_Number"].to_numpy()
    soh = battery["Predicted_SOH"].to_numpy()
    
    return cycle, soh

# %%

# %%

# %%
def degradation_fit(cycle, soh, window=20):
    '''Estimate the battery degradation trend using a linear regression. Only most recent cycles are used
    because recent trends are more indicative of battery's current behaviour.'''
    l = len(cycle)
    useful_cycle = []
    useful_soh = []
    for i in range(window):
        useful_cycle.append(cycle[l-window+i])
        useful_soh.append(soh[l-window+i])
    m,c = np.polyfit(useful_cycle, useful_soh, 1)
    cycle = np.array(cycle)
    soh = np.array(soh)
    cycle = cycle[-window:]
    soh = soh[-window:]

    predicted = m*cycle + c

    
    return m, c, r2_score(predicted, soh)

# %%
def estimate_rul(cycle, soh, threshold = 0.8):
    '''RUL(remaining Useful Life) is estimated using the slopes and intercepts calculated from 
    degradation fit curve. The battery end of life is taken at 0.8, consistent with standards.'''
    m,c, fit_r2 = degradation_fit(cycle,soh)
    failure_cycle = (threshold-c)/m #if m<0 else
    rul = failure_cycle-cycle[-1] if failure_cycle>cycle[-1] else 0
    curr_soh = soh[-1]
    return int(failure_cycle), int(rul), curr_soh, m, fit_r2


# %%

def health_category(soh, rul):
    '''Assign a health category using predicted soh and rul'''
    if soh<=0.8 or rul<=10:
        return 'Critical'
    elif soh<0.85 or rul<=30:
        return 'Warning'
    elif soh<0.9 or rul<=60:
        return 'Monitor'
    else: 
        return 'Healthy' 



# %%
def risk(soh, rul):
    '''calculate the risk score for a battery using soh and rul'''
    return (1-soh)/(rul+1)


# %%
def health_report(df):
    row=[]
    '''Generate battery health metrics from SOH predictions. For each battery this function estimates:
     - Current SOH
     - Failure cycle
     - Remaining Useful Life
     - Risk score
     - Health category
     - Confidence'''
    for battery_id in df['Battery_ID'].unique():
        cycle,soh=get_battery_history(df, battery_id)
        fc,rul,curr_soh,m, fit_r2 = estimate_rul(cycle,soh)
        rsk = risk(curr_soh, rul)
        category = health_category(curr_soh,rul)
       
        row.append({'Battery_ID': battery_id,
                    'Failure_cycle': fc,
                    'Remaining_useful_life': rul,
                  
                    'Current_SOH': curr_soh,
                   
                    'Confidence': fit_r2, 
                    'Degradation_rate': m,
                    'Health_Status': category,
                    'Risk': rsk})
    dm = pd.DataFrame(row)
    return dm


# %%


# %%



# 


# 


# 

# %%



