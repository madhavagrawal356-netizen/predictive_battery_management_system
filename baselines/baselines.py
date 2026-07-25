'''This script trains and evaluates Random Forest and XGBoost models
using handcrafted statistical features. It is intended for
experimental comparison with the CNN-Transformer model and is not
part of the deployment pipeline.'''
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import optuna
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')

# %%
# Load feature dataset
df = pd.read_csv('dataframe.csv')

# %%
df.dtypes
df['sampling_density']=df['sampling_density'].astype('float')

# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Train-test split
train_idx, test_idx = next(gss.split(df, groups=df['battery_id']))
train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

# %%
# Model definitions
rf = RandomForestRegressor(n_estimators=800, random_state=42)
xgb = XGBRegressor(n_estimators=800, random_state=42)

# %%
X_train = train_df.drop(columns=['soh', 'battery_id'])
y_train = train_df['soh']
X_test = test_df.drop(columns=['soh', 'battery_id'])
y_test = test_df['soh']
groups = train_df["battery_id"]



# %%
def models(trials, key):
    if key=='rf':
        rf = RandomForestRegressor(n_estimators=trials.suggest_int('n_estimators', 100, 1000), max_depth=trials.suggest_int('max_depth', 1, 20), min_samples_split=trials.suggest_int('min_samples_split', 2, 20), min_samples_leaf=trials.suggest_int('min_samples_leaf', 1, 20), random_state=42, n_jobs=-1)
        return rf
    else:
        xgb = XGBRegressor(n_estimators=trials.suggest_int('n_estimators', 100, 1000), max_depth=trials.suggest_int('max_depth', 1, 20), learning_rate=trials.suggest_float('learning_rate', 0.01, 0.3), subsample=trials.suggest_float('subsample', 0.5, 1.0), colsample_bytree=trials.suggest_float('colsample_bytree', 0.5, 1.0), random_state=42, n_jobs=-1)
        return xgb

# %%
def objective(trials, key, X, y, groups):
    gkf = GroupKFold(n_splits=5)
    model = models(trials,key)
    rmses=[]
    for train_idx, valid_idx in gkf.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid,pred))
        rmses.append(rmse)
    return np.mean(rmses)

    
    
    
    


# Hyperparameter tuning
def tune_model(key, X, y, groups, n_trials=10):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trials: objective(trials, key, X, y, groups), n_trials=n_trials)
    return study

# %%
rf_model = tune_model('rf', X_train, y_train, groups)
rf_model = RandomForestRegressor(**rf_model.best_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
print(f"Random Forest - R2: {r2:.4f}, MAPE: {MAPE:.4f}, MAE: {MAE:.4f}, MSE: {MSE:.4f}, RMSE: {RMSE:.4f}")

# %%
xgb_model = tune_model('xgb', X_train, y_train, groups)
xgb_model = XGBRegressor(**xgb_model.best_params, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
print(f"XGBoost : {r2:.4f}, MAPE: {MAPE:.4f}, MAE: {MAE:.4f}, MSE: {MSE:.4f}, RMSE: {RMSE:.4f}")

# %%
# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Capacity")
plt.ylabel("Predicted Capacity")
plt.title("Random Forest: Actual vs Predicted")

plt.tight_layout()
plt.savefig("rf_actual_vs_predicted.png", dpi=300)
plt.show()

# %%



