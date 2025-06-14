import pandas as pd
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv("data/monatszahlen2505_verkehrsunfaelle_06_06_25.csv")
    df = df[(df['MONATSZAHL']=='Alkoholunfälle') 
        & (df['AUSPRAEGUNG']=='insgesamt')]
    
    print("Unique MONAT values:", df['MONAT'].unique())
    df['month'] = df['MONAT'].astype(str).str[-2:]
    
    valid_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    df = df[df['month'].isin(valid_months)]
    
    df['date'] = pd.to_datetime(df['JAHR'].astype(str) + '-' + df['month'], format='%Y-%m')
    df = df.set_index('date')['WERT'].sort_index()
    print(df.head())