'''This module takes the loaded data and processes it to make python dictionaries useful in other parts of the code.'''
from .data_loader import load_all_batteries
import numpy as np


# %%

# %%


# %%
def get_cycles(battery):
    '''This function takes the loaded battery and extracts the cycle array which contains the cycles and their data.
    Input: loaded .mat dictionary
    Output: array containing every cycle and its data'''
    for key in battery.keys():
        if key.startswith("__")==False:
            battery_id = key
            break
    battery_struct = battery[battery_id][0,0]
    return battery_struct['cycle']


    

# %%
def is_discharge_cycle(cycle):
    '''checks if a cycle is a discharge cycle(as only discharge cycles are used for SOH calculation)
    Input: individual cycle array
    Output: True if a cycle is discharge cycle'''
    return cycle["type"][0]=='discharge'
    

# %%
def parse_cycle(cycle, battery_id, cycle_number):
    '''This function converts an individual discharge cycle to a python dictionary. In training datasets,capacity      is available and stored.'''
    data = cycle['data'][0,0]
    voltage = data['Voltage_measured'][0]
    current = data["Current_measured"][0]
    temperature = data["Temperature_measured"][0]
    capacity = None
    if 'Capacity' in data.dtype.names and len(data['Capacity'][0])>0:
        capacity=float(data["Capacity"][0][0])
    time =data['Time'][0]
    current_load = data['Current_load'][0]
    voltage_load = data['Voltage_load'][0]
    temp = cycle['ambient_temperature'][0]
    sample= {'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'time': time,
            'curr_load': current_load,
            'voltage_load': voltage_load,
            'amb_temp': temp,
            'battery_id': battery_id,
            'cycle_number': cycle_number}
    if capacity is not None:
        sample['capacity']=capacity
    return sample



# %%
def extract_discharge_cycles(battery, battery_id):
    '''This function extracts every discharge cycle from a battery
       Input: loaded battery and battery_id
       Output: list of parsed cycles in a battery'''
    cycles = get_cycles(battery)
    out = []
    for cycle_number, cycle in enumerate(cycles[0], start=1):
        if is_discharge_cycle(cycle):
            parsed=parse_cycle(cycle, battery_id, cycle_number)
            if parsed is not None:
                out.append(parsed)
    return out


# %%



