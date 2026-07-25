'''CNN-Transformer model for battery State of Health prediction.
The model first extracts local degradation patterns using 1D
convolutions and then models long-range temporal relationships
using Transformer encoder layers.'''
import torch.nn as nn
import torch
from data.dataloaders import create_dataloaders
from data.dataset import final_dataset
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np

# %%
class MainModel(nn.Module):
    """
    CNN-Transformer architecture for SOH prediction.

    Pipeline

    Input (6×256)

        ↓

    Conv1D + BatchNorm + ReLU

        ↓

    Conv1D + BatchNorm + ReLU

        ↓

    Positional Embedding

        ↓

    Transformer Encoder

        ↓

    Mean Pooling

        ↓

    Fully Connected Layers

        ↓

    SOH Prediction
    """
    def __init__(self, n_inputs):
        super(MainModel,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_inputs, out_channels=32, padding=1, kernel_size=3)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        self.bn1= nn.BatchNorm1d(32)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, padding=1, kernel_size=3)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        self.bn2= nn.BatchNorm1d(64)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.pos_embedding = nn.Parameter(torch.randn(1,64,64))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.encoderlayer1=nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoderlayer1, num_layers=2)
        self.fc1=nn.Linear(64,32)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        self.fc2=nn.Linear(32,1)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)
        self.dropout=nn.Dropout(0.2)
        self.cnn_layer=nn.Sequential(self.conv1, self.bn1, self.relu, self.pool, 
                                     self.conv2, self.bn2, self.relu, self.pool )
        
        
    def forward(self, x): #Input: (batch, 6, 256)
        out = self.cnn_layer(x)
        out = out.permute(0,2,1)
        out = out+self.pos_embedding
        out = self.transformer(out)
        out = out.mean(dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out





# %%
# Note: Uncomment the below lines to train the model again on a different data
'''sample = final_dataset(r'5. Battery Data Set\3. BatteryAgingARC_25-44')#in case training is needed again, add the path to the battery folder 

train_data, test_data, val_data=create_dataloaders(sample)
model=MainModel(6)'''


# %%


# %%
'''epochs = 10
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
def evaluate(val_data, model):
    model.eval()
    predictions=[]
    target=[]
    with torch.no_grad():
        for x,y,_,_ in val_data:
            yhat = model(x).squeeze(1)
            predictions.extend(yhat.cpu().numpy())
            target.extend(y.cpu().numpy())
    return r2_score(target,predictions), root_mean_squared_error(target, predictions)
        
best_rmse = float('inf')
counter = 10

for epoch in range(epochs):
    model.train()
    train_loss=0
    
    for x,y, _,_ in train_data:
        yhat=model(x).squeeze(1)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    train_loss /= len(train_data)
    r2, rmse = evaluate(val_data, model)
    if rmse<best_rmse:
        counter=10
        best_r2=r2
        best_rmse=rmse
        best_model_epoch=epoch
        torch.save(model.state_dict(), r'model\best_model.pth')
    else:
        counter-=1
    print ('epoch:', epoch+1)
    print ('r2:', r2)
    print ('rmse:' , rmse)
    print ('loss:', train_loss)
    print (f'Best model till now was epoch:{best_model_epoch+1} with r2:{best_r2}, rmse:{best_rmse}')


    if counter ==0:
        break'''
    


 



# %%
'''x, y = next(iter(val_data))

model.eval()
with torch.no_grad():
    pred = model(x).squeeze(1)

print(pred[:10])
print(y[:10])'''

# %%
'''model_best=MainModel(6)
best_model='best_model.pth'
state_dict= torch.load(best_model)
model_best.load_state_dict(state_dict)
model_best'''

# %%
'''r2, rmse = evaluate(test_data, model_best)
print(r2)
print(rmse)'''
            

# %%



