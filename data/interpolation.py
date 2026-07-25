'''Every discharge cycle is resampled to a fixed length so that all battery cycles have the same input dimensions for the neural network'''
import numpy as np

# %%
def interpolate_signal(signal, target_length=256):
    '''resample a one dimensional signal to a fixed length
    Input: array of signal
    Output: interpolated signal'''
    original_length= len(signal)
    if original_length == target_length:
        return signal
    else:
        old_x = np.linspace(0, 1, original_length)
        new_x = np.linspace(0, 1, target_length)
        return np.interp(new_x, old_x, signal)



# %%
def interpolate_cycle(cycle):
    '''convert one battery cycle into a model input format.
    Input: cycle containing different measurement channels
    Output: dictionary with (6,256) signal, battery_id and cycle number and capacity '''
    #interpolate each measurement
    voltage = interpolate_signal(cycle['voltage'])
    current = interpolate_signal(cycle['current'])
    
    temperature = interpolate_signal(cycle['temperature'])
    curr_load = interpolate_signal(cycle['curr_load'])
    voltage_load = interpolate_signal(cycle['voltage_load'])
    time = interpolate_signal(cycle['time'])
    signal = np.stack([voltage, current, temperature, curr_load, voltage_load, time], axis=0)#combine all channels
    sample= {"signal":signal,
            "battery_id":cycle['battery_id'],
            "cycle_number":cycle['cycle_number'],
            }
    if 'capacity' in cycle:  #retain capacity if it is present (training) otherwise leave it(evaluation)
        sample['capacity']=cycle['capacity']
    return sample
'''The model expects six input channels : Voltage, Current, Temperature,Current Load, Voltage Load and Time'''


# %%



# %%



