import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


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
    df = df.set_index('date').sort_index()
   
    return df


def create_prophet_model(df):
    """Create a Prophet model for forecasting"""
    prophet_df = df.reset_index()
    prophet_df = prophet_df.rename(columns={'date': 'ds', 'WERT': 'y'})
   
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
   
    model.fit(prophet_df)
   
    return model, prophet_df


def forecast_with_prophet(model, periods=12):
    """Create forecast using Prophet"""
    future = model.make_future_dataframe(periods=periods, freq='M')
   
    forecast = model.predict(future)
   
    return forecast


def visualize_prophet_forecast(model, forecast, df):
    """Create visualization with Prophet forecast"""
    fig1 = model.plot(forecast, figsize=(12, 8))
    plt.title('Alkoholunfälle - Prophet Forecast', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('alkohol_accidents_prophet_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
   
    fig2 = model.plot_components(forecast, figsize=(12, 10))
    plt.tight_layout()
    plt.savefig('alkohol_accidents_prophet_components.png', dpi=300, bbox_inches='tight')
    plt.show()
   
    jan_2021_forecast = forecast[forecast['ds'] == '2021-01-31']['yhat'].iloc[0]
   
    return jan_2021_forecast


def get_specific_forecast(forecast, target_date):
    """Get forecast for specific date"""
    target_forecast = forecast[forecast['ds'].dt.strftime('%Y-%m') == target_date]
    if not target_forecast.empty:
        return target_forecast.iloc[0]
    return None


if __name__ == "__main__":
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
    print(df.head())
   
    print("\nCreating Prophet forecasting model...")
    model, prophet_df = create_prophet_model(df)
   
    print("\nGenerating forecast...")
    forecast = forecast_with_prophet(model, periods=12)
   
    jan_2021_forecast_row = get_specific_forecast(forecast, '2021-01')
    jan_2021_forecast = jan_2021_forecast_row['yhat'] if jan_2021_forecast_row is not None else None
   
    print(f"Forecast for Alkoholunfälle in January 2021: {jan_2021_forecast:.0f} accidents")
    if jan_2021_forecast_row is not None:
        print(f"Confidence interval: [{jan_2021_forecast_row['yhat_lower']:.0f}, {jan_2021_forecast_row['yhat_upper']:.0f}]")
   
    print("\nCreating visualization...")
    visualized_forecast = visualize_prophet_forecast(model, forecast, df)
   
    print(f"\nSummary Statistics:")
    print(f"Historical average (2020): {df[df.index.year == 2020]['WERT'].mean():.1f}")
    print(f"January 2021 forecast: {jan_2021_forecast:.0f} accidents")
   
    print(f"\n2021 Forecast Summary:")
    forecast_2021 = forecast[forecast['ds'].dt.year == 2021][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_2021['month'] = forecast_2021['ds'].dt.strftime('%Y-%m')
    print(forecast_2021[['month', 'yhat', 'yhat_lower', 'yhat_upper']].round(0))