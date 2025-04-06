import pandas as pd
import pickle

#load the trained model
def load_model(model_path="src/ml model/xgb_model.pkl"):
    print('loading model...\n')
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

#load unseen data from a CSV file
def load_data(unseen_csv):
    print('loading data...\n')
    unseen_data = pd.read_csv(unseen_csv)
    return unseen_data

#data preprocessing
def preprocess_data(df):
    print('preprocessing...\n')
    #print("Columns before preprocessing:", df.columns)  #debug
    df = df[~df['Compound'].isin(['TEST_UNKNOWN', 'UNKNOWN', 'INTERMEDIATE', 'WET'])]
    df = df[(df['Delta'] <= 0.2) & (df['Delta'] >= -0.2)]
    df = df[df['TyreLife'] != df.groupby(['Session', 'Driver', 'Circuit', 'Year'])['TyreLife'].transform('min')]
    df = df.dropna(subset=['Compound'])
    #print("Columns after preprocessing:", df.columns)  #debug
    return df



def create_features(df_filtered):
    #print("Columns before feature creation:", df_filtered.columns)
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
    #print("Columns after feature creation:", df_filtered.columns)
    return df_filtered



def predict_with_model(model, df, num_f, cat_f):
    print('modelling the tyres...\n')
    df['Predicted_Delta'] = model.predict(df[num_f + cat_f])
    return df


def calculate_predicted_stats(df):
    print('calculating tyre data...\n')
    predicted_stats = {}
    for combo, group in df.groupby('Compound'):
        predicted_mean = group['Predicted_Delta'].mean()
        std = group['Predicted_Delta'].std()
        predicted_stats[combo] = {'Predicted Mean': predicted_mean, 'Predicted Std': std}
    return predicted_stats


def tyre_model(unseen_csv):
    model = load_model()
    unseen_data = load_data(unseen_csv)
    unseen_data = preprocess_data(unseen_data)
    unseen_data = create_features(unseen_data)
    #features
    num_f = ['LapNumber', 'LapTime_seconds', 'TyreLife', 'TrackTemp', 'Humidity', 'AirTemp', 'Stint', 
            'Prev_Time', 'Prev_Delta', 'RelativeDelta', 'RollingAvg_Time', 'RollingAvg_Delta',
            'TrackTempMean','AvgTyreLife','LapTime_TrackTemp','Delta_TrackTemp']
    cat_f = ['Compound']

    unseen_data = predict_with_model(model, unseen_data, num_f, cat_f)
    predicted_tyre_data = calculate_predicted_stats(unseen_data)

    return predicted_tyre_data
