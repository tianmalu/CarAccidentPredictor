import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath="data/monatszahlen2505_verkehrsunfaelle_06_06_25.csv"):
    """
    Load the CSV, filter for Alkoholunfälle & 'insgesamt',
    extract valid months, parse dates, and keep up to 2020.
    """
    df = pd.read_csv(filepath)
    df = df[(df['MONATSZAHL']=='Alkoholunfälle') & (df['AUSPRAEGUNG']=='insgesamt')]

    df['month'] = df['MONAT'].astype(str).str[-2:]
    valid_months = [f"{m:02d}" for m in range(1,13)]
    df = df[df['month'].isin(valid_months)]

    df['date'] = pd.to_datetime(df['JAHR'].astype(str) + '-' + df['month'], format='%Y-%m')
    df = df[df['JAHR'] <= 2020]
    df = df.set_index('date').sort_index()

    return df

def add_time_features(df):
    """
    Add:
      - t: months since start (integer sequence)
      - month_sin and month_cos: cyclical encoding of month-of-year
    """
    df_feat = df.copy()
    df_feat['t'] = np.arange(len(df_feat))
    df_feat['month_num'] = df_feat.index.month
    df_feat['month_sin'] = np.sin(2 * np.pi * (df_feat['month_num'] - 1) / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * (df_feat['month_num'] - 1) / 12)
    return df_feat

def prepare_target_features(df_model, target_year, target_month):
    """
    Build a single feature vector for a given (year, month):
      - t: months since the start of df_model up to this point
      - month_sin, month_cos as above
    """
    last_date = df_model.index[-1]
    months_ahead = (target_year - last_date.year) * 12 + (target_month - last_date.month)
    t_target = df_model['t'].iloc[-1] + months_ahead
    sin = np.sin(2 * np.pi * (target_month - 1) / 12)
    cos = np.cos(2 * np.pi * (target_month - 1) / 12)
    return [t_target, sin, cos]

def train_ridge_seasonal(df_model):
    """
    Use TimeSeriesSplit + GridSearchCV to find best alpha for Ridge,
    then return the trained model.
    """
    X = df_model[['t', 'month_sin', 'month_cos']].values
    y = df_model['WERT'].values

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    cv = GridSearchCV(Ridge(), param_grid, cv=tscv, scoring='neg_mean_squared_error')
    cv.fit(X, y)

    best = cv.best_estimator_
    print(f"Best alpha: {cv.best_params_['alpha']}")
    y_fit = best.predict(X)
    print(f"Train MSE: {mean_squared_error(y, y_fit):.2f}")
    return best

def forecast_periods(model, df_model, periods=12):
    """
    Generate a pandas.Series of length `periods` forecasts,
    with datetime index from the month after the last in df_model.
    """
    last = df_model.index[-1]
    future_index = pd.date_range(start=last + pd.offsets.MonthEnd(1),
                                 periods=periods, freq='M')
    feats = [prepare_target_features(df_model, d.year, d.month) for d in future_index]
    X_fut = np.array(feats)
    y_fut = model.predict(X_fut)
    return pd.Series(y_fut, index=future_index)

def visualize(df, forecast_series):
    """
    Plot historical WERT and forecast_series together.
    """
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['WERT'], label='Historical', linewidth=2)
    plt.plot(forecast_series.index, forecast_series.values, '--o',
             label='Forecast', linewidth=2)
    if '2021-01-31' in forecast_series.index:
        val = forecast_series.loc['2021-01-31']
        plt.plot(pd.to_datetime('2021-01-31'), val, 'ro',
                 label=f'Jan 2021: {val:.0f}')
    plt.title('Alkoholunfälle - Seasonal Ridge Regression Forecast')
    plt.xlabel('Date')
    plt.ylabel('Number of Accidents')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('seasonal_ridge_forecast.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} records from {df.index.min().date()} to {df.index.max().date()}")
    df_model = add_time_features(df)

    model = train_ridge_seasonal(df_model)

    fc = forecast_periods(model, df_model, periods=12)
    print(f"Forecast for Jan 2021: {fc.loc['2021-01-31']:.0f}")
    visualize(df, fc)
