'''This module converts the raw MTALAB battery files into a list of processed battery cycles to be used for training or inference.'''


from .parser import extract_discharge_cycles
from .interpolation import interpolate_cycle
from .data_loader import load_all_batteries



# %%
def build_batteries(batteries):
    '''extract all discharge cycles for all batteries
    Input: all_loaded batteries
    Output: dict{battery_id:[cycle1,cycle2.......]}'''
    out={}
    for battery_id, battery in batteries.items():
        out[battery_id]=extract_discharge_cycles(battery, battery_id)
    return out    
    


# %%
def interpolate_cycles(battery_dictionary):
    '''Interpolate everydischarged cycle to a fixed length
    Input: the processed batteries from build_batteries
    Output: processed battery dictionary with correct lengths'''
    out2={}
    for battery_id, cycles in battery_dictionary.items():
        for cycle in cycles:
            inerpolated_cycle = interpolate_cycle(cycle)
            out2.setdefault(battery_id, []).append(inerpolated_cycle)
    return out2


# %%
def build_capacity_lookup(battery_dictionary):
    capacity_lookup = {}
    '''Determine the initial battery capacity. The maximum observed discharge capacity is treated as the
reference (100% SOH) for that battery.'''
    for battery_id, battery in battery_dictionary.items():
        capacity_lookup[battery_id] = max(cycle['capacity'] for cycle in battery)
    return capacity_lookup

# %%
def add_soh(battery_dictionary, capacity_lookup):
    '''Compute State of Health (SOH).
    SOH = Current Capacity / Initial Capacity
    Output: battery dictionary with added soh for every cycle'''
    for battery_id, battery in battery_dictionary.items():
        max_capacity = capacity_lookup[battery_id]
        for cycle in battery:
            cycle['soh'] = cycle['capacity'] / max_capacity
    return battery_dictionary

# %%


# %%


# %%
def flatten_batteries(final_battery_dictionaries):
    '''Convert the nested battery dictionary into a flat list.'''
    out=[]
    for battery_id, battery in final_battery_dictionaries.items():
        for cycle in battery:
            out.append(cycle)
    return out


# %%
def final_dataset(file_path , training=True):
    '''build the full dataset.
    Input: folder having batteries
    Output: final flattened dataset having soh if training data == True'''
    batteries=load_all_batteries(file_path)# Load MATLAB files
    battery_dictionary=build_batteries(batteries)# Extract discharge cycles
    interpolated_dictionary=interpolate_cycles(battery_dictionary)# Interpolate every cycle
    if training:
        capacity_lookup=build_capacity_lookup(battery_dictionary)
        final_dictionary = add_soh(interpolated_dictionary, capacity_lookup)# Generate SOH labels only during training
    else:
        final_dictionary = interpolated_dictionary
    flattened=flatten_batteries(final_dictionary)# Convert dictionary into a flat dataset
    return flattened


# %%



