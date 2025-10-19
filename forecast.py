import pandas as pd
from prophet import Prophet
import streamlit as st

@st.cache_data
def train_and_forecast_models(file_path):
    """
    Loads data, cleans it, trains three separate Prophet models,
    and returns the trained models and their full forecast dataframes
    covering up to 2035.
    This function is cached to avoid re-training on every app interaction.
    """
    try:
        df = pd.read_csv(file_path, skiprows=1)

        # --- Robust Data Cleaning ---
        df['Reported Date'] = pd.to_datetime(df['Reported Date'], errors='coerce')

        price_cols = ['Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

        df_cleaned = df[['Reported Date'] + price_cols].dropna()

        # --- Train Models ---
        models = {}
        forecasts = {}

        for price_type in ['Modal', 'Max', 'Min']:
            price_col = f'{price_type} Price (Rs./Quintal)'

            df_prophet = df_cleaned[['Reported Date', price_col]].rename(columns={
                'Reported Date': 'ds',
                price_col: 'y'
            })

            model = Prophet()
            model.fit(df_prophet)

            # --- Forecast up to 2035 ---
            last_date_in_data = df_prophet['ds'].max()
            end_date = pd.Timestamp('2035-12-31')
            period_days = (end_date - last_date_in_data).days

            future = model.make_future_dataframe(periods=period_days)
            forecast = model.predict(future)

            models[price_type] = model
            forecasts[price_type] = forecast

        return models, forecasts

    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found. Please make sure it's in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return None, None
