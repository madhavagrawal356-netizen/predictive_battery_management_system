'''This module converts battery discharge cycles into a tabular
dataset by extracting handcrafted statistical features and
computing State of Health (SOH).'''
from .features import compute_soh, compute_current_features,compute_power_features,compute_temp_features,compute_time_features,compute_voltage_features
from data.parser import extract_discharge_cycles
import pandas as pd
from data.data_loader import load_all_batteries

# %%
def compute_features(cycle, initial_capacity):
    '''Compute all handcrafted features for a single discharge cycle.'''
    features={}
    features.update(compute_voltage_features(cycle['voltage']))
    features.update(compute_current_features(cycle['current']))
    features.update(compute_temp_features(cycle['temperature']))
    features.update(compute_time_features(cycle['time']))
    features.update(compute_power_features(cycle['voltage'], cycle['current'], cycle['time']))
    features['soh']=compute_soh(cycle['capacity'], initial_capacity)
    features['battery_id']=cycle['battery_id']
    return features

# %%
def battery_to_dataframe(battery, battery_id):
    '''Convert one battery into a feature DataFrame. Each row corresponds to one discharge cycle.'''
    cycles = extract_discharge_cycles(battery, battery_id)
    final=[]
    initial_capacity = max(cycle['capacity'] for cycle in cycles)
    for cycle in cycles:
        final.append(compute_features(cycle, initial_capacity))
    final = pd.DataFrame(final)
    return final


# %%
def make_dataset(folder_path):
    '''Generate a feature dataset for all batteries in a folder.'''
    batteries = load_all_batteries(folder_path)
    dfs=[]
    for battery_id, battery in batteries.items():
        dfs.append(battery_to_dataframe(battery, battery_id))
    return pd.concat(dfs, ignore_index=True)

# %%


# %%



