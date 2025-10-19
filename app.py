import streamlit as st
import pandas as pd
from datetime import date
from dotenv import load_dotenv
import os
from openai import OpenAI

from prophet.plot import plot_plotly, plot_components_plotly
from forecast import train_and_forecast_models

# --- Page Configuration ---
st.set_page_config(
    page_title="Garlic Price Forecaster & AI Analyst",
    page_icon="🌿",
    layout="wide"
)

# --- Load API Key ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found! Please add it to your .env file.", icon="🚨")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Load and Train Models (Cached) ---
DATA_FILE = 'garlic data for chat gpt2002 to 2025 (1).csv'
with st.spinner('Loading data and training forecasting models up to 2035...'):
    models, forecasts = train_and_forecast_models(DATA_FILE)

if models is None or forecasts is None:
    st.stop()

st.success('Forecast models trained successfully up to 2035!', icon="✅")

# --- Main App Interface ---
st.title("🧄 Garlic Price Forecaster & AI Analyst (2023-2035)")
st.markdown("Forecast garlic prices for the Gondal market from 2023 to 2035 and get AI-powered insights.")
st.warning("Long-term forecasts carry higher uncertainty. Use as trend guidance, not exact prices.", icon="⚠️")

# --- Create Tabs ---
tab1, tab2 = st.tabs(["📈 Price Forecasting", "🤖 Ask AI for Analysis"])

# --- Tab 1: Forecasting ---
with tab1:
    st.header("Select a Date to Forecast")

    # --- Date selection restricted to 2023-2035 ---
    min_date = pd.Timestamp('2023-01-01')
    max_date = pd.Timestamp('2035-12-31')

    future_date = st.date_input(
        "Select a future date:",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        help="Select a date between 2023 and 2035."
    )

    price_predictions = {}

    if future_date:
        st.subheader(f"🔮 Forecast for {future_date.strftime('%B %d, %Y')}")
        
        col1, col2, col3 = st.columns(3)
        for price_type in ['Modal', 'Min', 'Max']:
            forecast_df = forecasts[price_type]
            selected_date = pd.to_datetime(future_date)
            prediction = forecast_df[forecast_df['ds'] == selected_date]

            if not prediction.empty:
                predicted_price = prediction['yhat'].iloc[0]
                lower_bound = prediction['yhat_lower'].iloc[0]
                upper_bound = prediction['yhat_upper'].iloc[0]
                price_predictions[price_type] = predicted_price

                with eval(f"col{['Modal', 'Min', 'Max'].index(price_type) + 1}"):
                    st.metric(
                        label=f"Predicted {price_type} Price (Rs./Quintal)",
                        value=f"₹{predicted_price:,.2f}",
                        help=f"Expected range: ₹{lower_bound:,.2f} to ₹{upper_bound:,.2f}"
                    )
            else:
                 with eval(f"col{['Modal', 'Min', 'Max'].index(price_type) + 1}"):
                    st.warning(f"No prediction available for {price_type} price on this date.")

    st.divider()

    st.header("Interactive Forecast Plots")
    price_type_to_plot = st.selectbox(
        "Choose a price type to visualize:",
        ('Modal', 'Max', 'Min')
    )

    if price_type_to_plot:
        model_to_plot = models[price_type_to_plot]
        forecast_to_plot = forecasts[price_type_to_plot]

        st.subheader(f"Interactive Forecast for {price_type_to_plot} Price")
        fig_forecast = plot_plotly(model_to_plot, forecast_to_plot)
        fig_forecast.update_layout(
            title=f"{price_type_to_plot} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (Rs./Quintal)"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader(f"Forecast Components for {price_type_to_plot} Price")
        fig_components = plot_components_plotly(model_to_plot, forecast_to_plot)
        st.plotly_chart(fig_components, use_container_width=True)

# --- Tab 2: AI Analysis ---
with tab2:
    st.header("AI-Powered Market Analysis")
    st.markdown("Get insights from your selected forecast date.")

    if st.button("🤖 Analyze Forecast for Selected Date"):
        if not price_predictions:
            st.warning("Select a valid date in the first tab to get a forecast before analysis.")
        else:
            with st.spinner("AI Analyst is thinking..."):
                try:
                    prompt = f"""
You are a world-class agricultural market analyst for Gondal, Gujarat. Provide insights based on the garlic price forecast:

Forecast Date: {future_date.strftime('%B %d, %Y')}
Price Predictions (per Quintal):
- Min: ₹{price_predictions.get('Min', 0):,.2f}
- Modal: ₹{price_predictions.get('Modal', 0):,.2f}
- Max: ₹{price_predictions.get('Max', 0):,.2f}

Instructions:
1. Acknowledge long-term forecast uncertainty.
2. Summarize the forecast.
3. Explain the significance of price range.
4. Provide actionable advice for farmers/traders.
5. Use simple language.
6. Format using Markdown headings.
"""
                    response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[
                            {"role": "system", "content": "You are a helpful agricultural market analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                    )
                    ai_response = response.choices[0].message.content
                    st.markdown(ai_response)

                except Exception as e:
                    st.error(f"An error occurred while contacting the AI model: {e}")
