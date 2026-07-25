'''This module wraps the processed battery samples into PyTorch Dataset
objects and creates DataLoaders for training, validation, testing,
and inference.'''
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import GroupShuffleSplit 



# %%


# %%
class BatteryDataset(Dataset):
    '''Dataset used during model training.

    Each sample contains:
        - signal
        - SOH label
        - battery ID
        - cycle number'''
    def __init__(self, battery_data):
        self.battery_data = battery_data

    def __len__(self):
        return len(self.battery_data)

    def __getitem__(self, idx):
        cycle = self.battery_data[idx]
        signal = torch.tensor(cycle['signal'], dtype=torch.float32)#convert signal to a pytorch tensor
        soh = torch.tensor(cycle['soh'], dtype=torch.float32)

        return signal, soh, cycle['battery_id'], cycle['cycle_number']

class InferenceDataset(Dataset):
    '''Dataset used during deployment.

    Since the true SOH is unknown, only the input signal and
    metadata are returned.'''
    def __init__(self, battery_data):
        self.battery_data = battery_data
    def __len__(self):
        return len(self.battery_data)
    def __getitem__(self, idx):
        cycle = self.battery_data[idx]
        signal = torch.tensor(cycle['signal'], dtype=torch.float32)
        return signal, cycle['battery_id'], cycle['cycle_number']

# %%
def create_dataloaders(sample, batch_size=32, test_size=0.2, random_state=42):
   
   '''Split the dataset into training, validation, and testing sets.Batteries are split using GroupShuffleSplit so that cycles from the
       same battery never appear in multiple sets.'''
   gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
   gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)

    
   groups = [cycle['battery_id'] for cycle in sample]

   
   train_idx, test_idx = next(gss1.split(sample, groups=groups))
   

    
   train_data = [sample[i] for i in train_idx]
   temp_data = [sample[i] for i in test_idx]
   temp_groups = [cycle["battery_id"] for cycle in temp_data]
   test_idx, val_idx = next(gss2.split(temp_data, groups=temp_groups))
   test_data= [temp_data[i] for i in test_idx]
   val_data = [temp_data[i] for i in val_idx]

    
   train_loader = DataLoader(BatteryDataset(train_data), batch_size=batch_size, shuffle=True)
   test_loader = DataLoader(BatteryDataset(test_data), batch_size=batch_size, shuffle=False)
   val_loader = DataLoader(BatteryDataset(val_data), batch_size=batch_size, shuffle=False)

   return train_loader, test_loader, val_loader

# %%
#sample=final_dataset(r'5. Battery Data Set\3. BatteryAgingARC_25-44')


# %%
#train, test, val=create_dataloaders(sample)

# %%

# %%



