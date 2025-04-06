import gc #garbage colection
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('data/train/all_ferrari_22_23.csv')

df_filtered = df[
    (df['TyreLife'] != df.groupby(['Session', 'Driver', 'Stint', 'Circuit', 'Year'])['TyreLife'].transform('min')) & 
    (df['Delta'].between(-0.2, 0.2)) &
    (~df['Compound'].isin(['TEST_UNKNOWN', 'UNKNOWN', 'WET', 'INTERMEDIATE']))
].copy()

df_filtered = df_filtered.dropna(subset=['Compound'])
gc.collect()

def create_features(df_filtered):
    window_size = 5
    df_filtered['RelativeDelta'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound'])['Delta'].shift(1).diff()
    df_filtered['LapTime_TrackTemp'] = df_filtered['Delta'].shift(1) / (df_filtered['TrackTemp'].shift(1))
    df_filtered['Delta_TrackTemp'] = df_filtered.groupby(['Circuit', 'Compound'])['TrackTemp'].shift(1).diff()
    df_filtered['AvgTyreLife'] = df_filtered.groupby(['Year', 'Session', 'Compound', 'Driver','Circuit'])['TyreLife'].transform('mean')
    df_filtered['TrackTempMean'] = df_filtered.groupby('Circuit')['TrackTemp'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df_filtered['Prev_Time'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound'])['LapTime_seconds'].shift(1)
    df_filtered['Prev_Delta'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound'])['Delta'].shift(1)
    df_filtered['RollingAvg_Time'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound', 'Stint'])['LapTime_seconds'].shift(1).rolling(window=window_size, min_periods=3).mean()
    df_filtered['RollingAvg_Delta'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound', 'Stint'])['Delta'].shift(1).rolling(window=window_size, min_periods=3).mean()
    df_filtered = df_filtered.dropna()
    
    return df_filtered

#features
num_f = ['LapNumber', 'LapTime_seconds', 'TyreLife', 'TrackTemp', 'Humidity', 'AirTemp', 'Stint', 
         'Prev_Time', 'Prev_Delta', 'RelativeDelta', 'RollingAvg_Time', 'RollingAvg_Delta',
         'TrackTempMean','AvgTyreLife','LapTime_TrackTemp','Delta_TrackTemp']
cat_f = ['Compound']

gc.collect()

#traning
df_train = create_features(df_filtered)
#===============================================PARAMS===============================================
"""
Tested with optuna, these are baseline parameters
trial 1
[I 2025-03-31 04:45:07,554] Trial 558 finished with value: 0.18121137385052133 and 
parameters: {'n_estimators': 10841, 'max_depth': 81, 'min_child_weight': 2, 
'learning_rate': 0.04378630174606741, 'subsample': 1.0, 'colsample_bytree': 0.9310241226721427, 
'alpha': 3.428664884445604, 'lambda': 1.8114819826073918, 'gamma': 0.00016321886524062334}.
Best is trial 558 with value: 0.18121137385052133.

trial 2
[I 2025-03-31 21:59:33,239] Trial 70 finished with value: 0.20909320583787416 and 
parameters: {'n_estimators': 31541, 'max_depth': 84, 'min_child_weight': 47, 'learning_rate': 0.03880469099763725, 
'colsample_bytree': 0.9122646821577481, 'alpha': 0.01324854298149497, 'lambda': 0.6662036242437757, 
'gamma': 0.005518222727177145}. Best is trial 70 with value: 0.20909320583787416.

trial3 
I 2025-03-31 22:27:04,535] Trial 71 finished with value: 0.2337499697476194 and 
parameters: {'n_estimators': 35077, 'max_depth': 82, 'min_child_weight': 48, 
'learning_rate': 0.02982483302159192, 'colsample_bytree': 0.9125971646738913, 'alpha': 0.0510371212313776, 'lambda': 0.7043426701732907, 
'gamma': 0.005250681739806855}. Best is trial 71 with value: 0.2337499697476194.

[I 2025-04-01 12:53:41,075] Trial 11 finished with value: 0.11703468436512943 and 
parameters: {'n_estimators': 17091, 'max_depth': 95, 'min_child_weight': 2, 'learning_rate': 0.001385330386387918, 
'colsample_bytree': 0.9979845568532735, 'alpha': 0.4275203206503395, 'lambda': 0.18127377264544312, 
'gamma': 0.0010320906645723202}. Best is trial 11 with value: 0.11703468436512943.
"""

xgb_params = {
    'n_estimators': 200,
    'max_depth': 125,
    'min_child_weight': 47,
    'learning_rate': 0.03880469099763725,
    'subsample': 1,
    'colsample_bytree': 1,
    'alpha': 0.01324854298149497,
    'lambda': 0.6662036242437757,
    'gamma':0.005250681739806855,
    'tree_method': 'hist', #ensure ml model uses all available resouces
    'random_state': 42,
    'n_jobs': -1
}

final_model = XGBRegressor(**xgb_params)
#===============================================PREPROCESS===============================================
preprocessor = ColumnTransformer([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'), cat_f),
    ('robust', RobustScaler(), ['LapTime_seconds', 'RollingAvg_Time', 'RollingAvg_Delta', 
                                 'Prev_Time', 'Prev_Delta', 'RelativeDelta','LapTime_TrackTemp']),
    ('standard', StandardScaler(), ['TrackTempMean', 'TrackTemp', 'AirTemp', 'Humidity','Delta_TrackTemp']),
    ('passthrough', 'passthrough', ['LapNumber', 'TyreLife', 'Stint', 'AvgTyreLife'])
])

#===============================================PIPELIEN===============================================
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', final_model)
])
#===============================================TRAINING===============================================
xgb_pipeline.fit(df_train[num_f + cat_f], df_train['Delta'])
train_preds = xgb_pipeline.predict(df_train[num_f + cat_f])

train_mse = mean_squared_error(df_train['Delta'], train_preds)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(df_train['Delta'], train_preds)
train_r2 = r2_score(df_train['Delta'], train_preds)

print(f'Training MSE: {train_mse:.4f}')
print(f'Training RMSE: {train_rmse:.4f}')
print(f'Training MAE: {train_mae:.4f}')
print(f'Training R2 Score: {train_r2:.4f}')
#===============================================TEST SET===============================================
#test
df_ferrari = pd.read_csv('data/test/ferrari_australia_sessions.csv')
df_ferrari_filtered = df_ferrari[
    (df_ferrari['TyreLife'] != df_ferrari.groupby(['Session', 'Driver', 'Stint', 'Circuit', 'Year'])['TyreLife'].transform('min')) & 
    (df_ferrari['Delta'] >= -0.2) & (df_ferrari['Delta'] <= 0.2) &
    (~df_ferrari['Compound'].isin(['TEST_UNKNOWN', 'UNKNOWN', 'WET', 'INTERMEDIATE']))
].copy()

df_ferrari_filtered = df_ferrari_filtered.dropna(subset=['Compound'])

df_ferrari_filtered = create_features(df_ferrari_filtered)
df_ferrari_filtered['Predicted_Delta'] = xgb_pipeline.predict(df_ferrari_filtered[num_f + cat_f])

#===============================================METRICS===============================================
test_mse = mean_squared_error(df_ferrari_filtered['Delta'], df_ferrari_filtered['Predicted_Delta'])
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(df_ferrari_filtered['Delta'], df_ferrari_filtered['Predicted_Delta'])
test_r2 = r2_score(df_ferrari_filtered['Delta'], df_ferrari_filtered['Predicted_Delta'])

print(f'Test MSE: {test_mse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test MAE: {test_mae:.4f}')
print(f'Test R2 Score: {test_r2:.4f}')

#===============================================VIS===============================================
#actal v pred
plt.figure(figsize=(10, 6))
plt.scatter(df_ferrari_filtered['Delta'], df_ferrari_filtered['Predicted_Delta'], alpha=0.3)
plt.plot([-0.2, 0.2], [-0.2, 0.2], color='red', linestyle='--')
plt.xlabel('Actual Delta')
plt.ylabel('Predicted Delta')
plt.title('Actual vs Predicted Delta')
plt.show()

#residula plot
residuals = df_ferrari_filtered['Delta'] - df_ferrari_filtered['Predicted_Delta']
plt.figure(figsize=(10, 6))
plt.scatter(df_ferrari_filtered['Predicted_Delta'], residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Residuals')
plt.ylabel('Predicted Delta')
plt.title('Residual Plot')
plt.show()

#residula dst
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.show()

#comparison
plt.figure(figsize=(10, 5))
plt.hist(df_ferrari_filtered['Delta'], bins=30, alpha=0.5, label='Actual Delta', color='blue', edgecolor='black')
plt.hist(df_ferrari_filtered['Predicted_Delta'], bins=30, alpha=0.5, label='Predicted Delta', color='orange', edgecolor='black')
plt.xlabel('Delta')
plt.ylabel('Frequency')
plt.title('Actual vs Predicted Delta Distribution')
plt.legend()
plt.show()

#save model
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_pipeline, f)