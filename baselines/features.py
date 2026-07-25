'''This module was used to make features to train benchmark models to be compared against 
the deep learning model. This module is currently not used.'''
import numpy as np

# %%
def compute_soh(capacity, initial_capacity):
    return capacity/initial_capacity

# %%
def compute_voltage_features(voltage_list):
    '''Compute statistical features from the voltage signal.
    Input: the voltage signal
    Output: the Mean, max, min, standard deviation, RMS, and range of voltage.'''
    mean= np.mean(voltage_list)
    max = np.max(voltage_list)
    min = np.min(voltage_list)
    std_dev = np.std(voltage_list)
    voltage_rms = np.sqrt(np.mean(voltage_list**2))
    range = max-min
    return {
        'volt_mean': mean,
        'volt_max': max,
        'volt_min': min,
        'volt_std': std_dev,
        'volt_rms': voltage_rms,
        'volt_range': range
    }


# %%
def compute_current_features(current_list):
    '''Compute statistical features from the current signal.
        Input: the current signal
        Output: the Mean, max, min, standard deviation, RMS of current.'''
    mean= np.mean(current_list)
    max = np.max(current_list)
    min = np.min(current_list)
    std_dev = np.std(current_list)
    curr_rms = np.sqrt(np.mean(current_list**2))
    return {
        'curr_mean': mean,
        'curr_max': max,
        'curr_min': min,
        'curr_std': std_dev,
        'curr_rms': curr_rms,
    }


# %%
def compute_temp_features(temp_list):
    '''Compute statistical features from the temperature signal.
        Input: the temperature signal
            Output: the Mean, max, min, standard deviation, rise in temperature.'''
    mean= np.mean(temp_list)
    max = np.max(temp_list)
    min = np.min(temp_list)
    std_dev = np.std(temp_list)
    temp_rise = temp_list[-1]-temp_list[0]
    return {
        'temp_mean': mean,
        'temp_max': max,
        'temp_min': min,
        'temp_std': std_dev,
        'temp_rms': temp_rise,
    }

# %%
def compute_time_features(time_list):
    '''Compute the duration and number of samples'''
    dur = time_list[-1]-time_list[0]
    sampling = len(time_list)
    return {
        'duration': dur,
        'sampling_density': sampling
    }

# %%
def compute_power_features(voltage_list, current_list, time_list):
    '''Compute power features'''
    power = [voltage_list[i]*current_list[i] for i in range(len(voltage_list))]
    energy = np.trapz(power, time_list)
    return {
        'power_mean':np.mean(power),
        'power_max':np.max(power),
        'power_std': np.std(power),
        'energy':energy
    }

# %%



