import pandas as pd
from prophet import Prophet
import joblib

def load_and_prepare_data():
    df = pd.read_csv("../data/monatszahlen2505_verkehrsunfaelle_06_06_25.csv")
    df = df[(df['MONATSZAHL']=='Alkoholunfälle')
        & (df['AUSPRAEGUNG']=='insgesamt')]
   
    print("Unique MONAT values:", df['MONAT'].unique())
   
    df['month'] = df['MONAT'].astype(str).str[-2:]
   
    valid_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    df = df[df['month'].isin(valid_months)]
   
    df['date'] = pd.to_datetime(df['JAHR'].astype(str) + '-' + df['month'], format='%Y-%m')
    df = df[df['JAHR'] <= 2020]
    
    prophet_df = df[['date', 'WERT']].copy()
    prophet_df = prophet_df.rename(columns={'date': 'ds', 'WERT': 'y'})
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
   
    return prophet_df

prophet_df = load_and_prepare_data()
print(f"Prophet data shape: {prophet_df.shape}")
print("First few rows:")
print(prophet_df.head())

model = Prophet(yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative')
model.fit(prophet_df)

joblib.dump(model, "model_prophet.joblib")
last_date = prophet_df['ds'].max()
joblib.dump(last_date, "last_date.joblib")

print(f"Model saved successfully!")
print(f"Last training date: {last_date}")