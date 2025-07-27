import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load your dataset (same one used for training)
df = pd.read_csv('DailyBeijingClimate.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Select same features
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
df = df[features]

# Fit scalers and save them
scalers = {}
for column in df.columns:
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    scalers['scaler_' + column] = scaler
    joblib.dump(scaler, f'scaler_{column}.save')  # Save to file

print("✅ All scalers saved successfully.")
