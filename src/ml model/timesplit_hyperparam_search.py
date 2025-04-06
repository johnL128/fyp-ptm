import gc #garbage colection
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import optuna

#load
df = pd.read_csv('data/train/all_ferrari_22_23.csv')
ferrari_val_data = pd.read_csv('data/test/ferrari_australia_sessions.csv')
df_ferrari_test = pd.read_csv('data/test/ferrari_china_sessions.csv')


#filter
df_filtered = df[
    (df['TyreLife'] != df.groupby(['Session', 'Driver', 'Stint', 'Circuit', 'Year'])['TyreLife'].transform('min')) & 
    (df['Delta'].between(-0.2, 0.2)) &
    (~df['Compound'].isin(['TEST_UNKNOWN', 'UNKNOWN', 'WET', 'INTERMEDIATE']))
].copy()
df_filtered = df_filtered.dropna(subset=['Compound'])


ferrari_val_data = ferrari_val_data[
    (ferrari_val_data['TyreLife'] != ferrari_val_data.groupby(['Session', 'Driver', 'Stint', 'Circuit', 'Year'])['TyreLife'].transform('min')) & 
    (ferrari_val_data['Delta'] >= -0.2) & (ferrari_val_data['Delta'] <= 0.2) &
    (~ferrari_val_data['Compound'].isin(['TEST_UNKNOWN', 'UNKNOWN', 'WET', 'INTERMEDIATE']))
].copy()
ferrari_val_data = ferrari_val_data.dropna(subset=['Compound'])

df_ferrari_test = df_ferrari_test[
    (df_ferrari_test['TyreLife'] != df_ferrari_test.groupby(['Session', 'Driver', 'Stint', 'Circuit', 'Year'])['TyreLife'].transform('min')) & 
    (df_ferrari_test['Delta'] >= -0.2) & (df_ferrari_test['Delta'] <= 0.2) &
    (~df_ferrari_test['Compound'].isin(['TEST_UNKNOWN', 'UNKNOWN', 'WET', 'INTERMEDIATE']))
].copy()
df_ferrari_test = df_ferrari_test.dropna(subset=['Compound'])

gc.collect()

#===============================================FETURE CREATION===============================================
def create_features(df_filtered):
    window = 5
    df_filtered['RelativeDelta'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound'])['Delta'].shift(1).diff()
    df_filtered['LapTime_TrackTemp'] = df_filtered['Delta'].shift(1) / (df_filtered['TrackTemp'].shift(1))
    df_filtered['Delta_TrackTemp'] = df_filtered.groupby(['Circuit', 'Compound'])['TrackTemp'].shift(1).diff()
    df_filtered['AvgTyreLife'] = df_filtered.groupby(['Year', 'Session', 'Compound', 'Driver','Circuit'])['TyreLife'].transform('mean')
    df_filtered['TrackTempMean'] = df_filtered.groupby('Circuit')['TrackTemp'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    df_filtered['Prev_Time'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound'])['LapTime_seconds'].shift(1)
    df_filtered['Prev_Delta'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound'])['Delta'].shift(1)
    df_filtered['RollingAvg_Time'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound', 'Stint'])['LapTime_seconds'].shift(1).rolling(window=window, min_periods=3).mean()
    df_filtered['RollingAvg_Delta'] = df_filtered.groupby(['Session', 'Driver', 'Circuit', 'Compound', 'Stint'])['Delta'].shift(1).rolling(window=window, min_periods=3).mean()
    df_filtered = df_filtered.dropna()
    return df_filtered

num_f = ['LapNumber', 'LapTime_seconds', 'TyreLife', 'TrackTemp', 'Humidity', 'AirTemp', 'Stint', 
         'Prev_Time', 'Prev_Delta', 'RelativeDelta', 'RollingAvg_Time', 'RollingAvg_Delta',
         'TrackTempMean','AvgTyreLife','LapTime_TrackTemp','Delta_TrackTemp']
cat_f = ['Compound']

gc.collect()

#===============================================PREPROCESS===============================================
def preprocess():
    return ColumnTransformer([
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), cat_f),
        ('robust', RobustScaler(), ['LapTime_seconds', 'RollingAvg_Time', 'RollingAvg_Delta', 
                                     'Prev_Time', 'Prev_Delta', 'RelativeDelta','LapTime_TrackTemp']),
        ('standard', StandardScaler(), ['TrackTempMean', 'TrackTemp', 'AirTemp', 'Humidity','Delta_TrackTemp']),
        ('passthrough', 'passthrough', ['LapNumber', 'TyreLife', 'Stint', 'AvgTyreLife'])
    ])

#===============================================PIPIENLINE===============================================
def get_xgb_pipeline(params):
    xgb_model = XGBRegressor(random_state=42, **params)
    return Pipeline([
        ('preprocessor', preprocess()),
        ('xgb', xgb_model)
    ])

#===============================================LOOK FOR PARAMS USING OPTUNA FUNCTOIN===============================================
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 50000),
        'max_depth': trial.suggest_int('max_depth', 2, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05),
        'subsample': 1.0,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.82, 1.0),
        'alpha': trial.suggest_float('alpha', 0, 10),
        'lambda': trial.suggest_float('lambda', 0, 20),
        'gamma': trial.suggest_float('gamma', 0, 0.01),
        'tree_method': 'hist',
        'n_jobs': -1
    }

    train_data = create_features(df_filtered.copy())
    val_data = create_features(ferrari_val_data.copy())
    
    pipeline = get_xgb_pipeline(params)
    pipeline.fit(train_data[num_f + cat_f], train_data['Delta'])
    
    val_preds = pipeline.predict(val_data[num_f + cat_f])
    return r2_score(val_data['Delta'], val_preds)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
"""
TPE sampler is used as it learns from past trials by modelling good and bad results separately,
then picks new parameters that look more like the good ones.
"""
study.optimize(objective, n_trials=500)

best_params = study.best_trial.params
xgb_pipeline = get_xgb_pipeline(best_params)
df_train = create_features(df_filtered.copy())
xgb_pipeline.fit(df_train[num_f + cat_f], df_train['Delta'])

#===============================================MODEL EVALUATION===============================================
def evaluate_model(df, dataset_name):
    df = create_features(df.copy())
    df['Predicted_Delta'] = xgb_pipeline.predict(df[num_f + cat_f])
    r2 = r2_score(df['Delta'], df['Predicted_Delta'])
    mae = mean_absolute_error(df['Delta'], df['Predicted_Delta'])
    mse = mean_squared_error(df['Delta'], df['Predicted_Delta'])
    rmse = np.sqrt(mse)
    print(f"\n{dataset_name} Metrics:")
    print(f"R²: {r2}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

evaluate_model(ferrari_val_data, "Validation (Ferrari Australian GP)")
evaluate_model(df_ferrari_test, "Test (Ferrari China GP)")
