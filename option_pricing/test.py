import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from datetime import datetime, timedelta
from math import log, sqrt, exp
from scipy.stats import norm
import yfinance as yf

from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel


# -------------------- Enums --------------------
class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Model'


# -------------------- Cache & Data --------------------
@st.cache_data
def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Charge les données depuis ./CSVs si dispo, sinon télécharge via yfinance puis les cache."""
    os.makedirs("./CSVs", exist_ok=True)
    path = f'./CSVs/{ticker}_returns.csv'
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(how="any", inplace=True)
        if df.empty:
            raise ValueError("Cached CSV is empty after cleaning.")
    except Exception as e:
        st.warning(f"Downloading data for {ticker} due to error: {e}")
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            df.to_csv(path)
    return df

def get_current_price(ticker: str):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return float(data['Close'].iloc[-1])
    except Exception as e:
        st.error(f"Error fetching current price for {ticker}: {str(e)}")
        return None


# -------------------- UI helpers --------------------
def get_common_inputs(default_ticker: str = "AAPL"):
    ticker = st.text_input('Ticker symbol', default_ticker)
    st.caption("Enter the stock symbol (e.g., AAPL for Apple Inc.)")
    current_price = get_current_price(ticker)

    if current_price is not None:
        st.write(f"Current price of {ticker}: ${current_price:.2f}")
        min_strike = max(0.1, round(current_price * 0.5, 2))
        max_strike = round(current_price * 2, 2)
        default_strike = round(current_price, 2)
    else:
        min_strike, max_strike, default_strike = 0.01, 1000.0, 100.0
        st.warning("Unable to fetch current price. Using default strike range.")

    strike_price = st.number_input(
        'Strike price',
        min_value=min_strike,
        max_value=max_strike,
        value=default_strike,
        step=0.01
    )
    return ticker, strike_price


# -------------------- Black-Scholes helpers (prix + greeks) --------------------
def _d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, r, sigma, T):
    return _d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

def bs_call_put(S, K, r, sigma, T):
    d1 = _d1(S, K, r, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(call), float(put)


# -------------------- App --------------------
st.title('Option Pricing')

pricing_method = st.sidebar.radio(
    'Please select option pricing model',
    options=[model.value for model in OPTION_PRICING_MODEL]
)
st.subheader(f'Pricing method: {pricing_method}')


# -------------------- BLACK-SCHOLES --------------------
if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
    # Layout: inputs à gauche, résultats/plots à droite
    left, right = st.columns([1, 2], vertical_alignment="top")

    with left:
        ticker, strike_price = get_common_inputs()
        risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10) / 100.0
        st.caption("Theoretical risk-free return (annualized).")
        sigma = st.slider('Sigma (Volatility) (%)', 0, 100, 20) / 100.0
        st.caption("Annualized volatility.")
        exercise_date = st.date_input(
            'Exercise date',
            min_value=datetime.today().date() + timedelta(days=1),
            value=datetime.today().date() + timedelta(days=365)
        )
        st.caption("The date when the option can be exercised.")
        run = st.button(f'Calculate option price for {ticker}')

    with right:
        tab_prices, tab_heatmap = st.tabs(["Prices", "Heatmap"])

    if run:
        try:
            with st.spinner('Fetching data...'):
                data = get_stock_data(
                    ticker,
                    start='2020-01-01',
                    end=datetime.today().strftime('%Y-%m-%d')
                )

            if data is None or data.empty:
                st.error("Unable to proceed with calculations due to data fetching error.")
            else:
                spot_price = float(data['Close'].iloc[-1])
                days_to_maturity = (exercise_date - datetime.now().date()).days
                if days_to_maturity <= 0:
                    st.error("Exercise date must be in the future.")
                else:
                    # --- PRICES (affiché au même endroit, en haut à droite) ---
                    T = days_to_maturity / 365.0
                    call_price, put_price = bs_call_put(
                        spot_price, strike_price, risk_free_rate, sigma, T
                    )

                    with right:
                        with tab_prices:
                            c1, c2 = st.columns(2)
                            c1.metric("Call price", f"{call_price:,.2f}")
                            c2.metric("Put price",  f"{put_price:,.2f}")

                            st.caption("Spot/inputs used:")
                            st.write(
                                pd.DataFrame(
                                    {
                                        "Spot": [round(spot_price, 4)],
                                        "Strike": [round(strike_price, 4)],
                                        "r": [risk_free_rate],
                                        "σ": [sigma],
                                        "T (years)": [round(T, 6)],
                                    }
                                )
                            )

                        # --- HEATMAPS (Spot × Vol, K fixé) ---
                        with tab_heatmap:
                            st.markdown("#### Heatmap parameters")
                            colA, colB, colC = st.columns(3)
                            with colA:
                                s_min = st.number_input("Min Spot", value=max(1.0, round(spot_price * 0.8, 2)))
                                v_min = st.number_input("Min Vol (σ)", value=max(0.05, round(sigma * 0.5, 2)))
                            with colB:
                                s_max = st.number_input("Max Spot", value=round(spot_price * 1.2, 2))
                                v_max = st.number_input("Max Vol (σ)", value=round(min(1.0, sigma * 1.5), 2))
                            with colC:
                                nS = st.slider("Spot grid", 10, 60, 21)
                                nV = st.slider("Vol grid", 10, 60, 21)

                            Ss = np.linspace(s_min, s_max, nS)
                            Vs = np.linspace(v_min, v_max, nV)
                            call_grid = np.zeros((nV, nS))
                            put_grid = np.zeros((nV, nS))

                            for i, v in enumerate(Vs):
                                for j, s in enumerate(Ss):
                                    c, p = bs_call_put(s, strike_price, risk_free_rate, v, T)
                                    call_grid[i, j] = c
                                    put_grid[i, j] = p

                            st.markdown("##### Call Price Heatmap")
                            fig1, ax1 = plt.subplots()
                            im1 = ax1.imshow(
                                call_grid, origin="lower", aspect="auto",
                                extent=[Ss.min(), Ss.max(), Vs.min(), Vs.max()]
                            )
                            ax1.set_xlabel("Spot Price"); ax1.set_ylabel("Volatility (σ)")
                            plt.colorbar(im1, ax=ax1)
                            st.pyplot(fig1)

                            st.markdown("##### Put Price Heatmap")
                            fig2, ax2 = plt.subplots()
                            im2 = ax2.imshow(
                                put_grid, origin="lower", aspect="auto",
                                extent=[Ss.min(), Ss.max(), Vs.min(), Vs.max()]
                            )
                            ax2.set_xlabel("Spot Price"); ax2.set_ylabel("Volatility (σ)")
                            plt.colorbar(im2, ax=ax2)
                            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")
    else:
        st.info("Click 'Calculate option price' to see results.")


# -------------------- MONTE CARLO (inchangé) --------------------
elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    ticker, strike_price = get_common_inputs()
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10) / 100.0
    st.caption("Theoretical risk-free return (annualized).")
    sigma = st.slider('Sigma (Volatility) (%)', 0, 100, 20) / 100.0
    st.caption("Annualized volatility.")
    exercise_date = st.date_input(
        'Exercise date',
        min_value=datetime.today().date() + timedelta(days=1),
        value=datetime.today().date() + timedelta(days=365)
    )
    st.caption("The date when the option can be exercised.")

    number_of_simulations = st.slider('Number of simulations', 100, 100000, 10000)
    st.caption("The number of price paths to simulate. More simulations increase accuracy but take longer to compute.")

    max_lines = max(1, int(number_of_simulations / 10))
    num_of_movements = st.slider(
        'Number of price movement simulations to be visualized', 1, max_lines, min(100, max_lines)
    )
    st.caption("The number of simulated price paths to display on the graph.")

    if st.button(f'Calculate option price for {ticker}'):
        try:
            with st.spinner('Fetching data...'):
                data = get_stock_data(
                    ticker,
                    start='2020-01-01',
                    end=datetime.today().strftime('%Y-%m-%d')
                )

            if data is not None and not data.empty:
                st.write("Data fetched successfully:")
                st.dataframe(data.tail())

                spot_price = float(data['Close'].iloc[-1])
                days_to_maturity = (exercise_date - datetime.now().date()).days
                if days_to_maturity <= 0:
                    st.error("Exercise date must be in the future.")
                else:
                    MC = MonteCarloPricing(
                        underlying_spot_price=spot_price,
                        strike_price=strike_price,
                        days_to_maturity=days_to_maturity,
                        risk_free_rate=risk_free_rate,
                        sigma=sigma,
                        number_of_simulations=number_of_simulations
                    )

                    MC.simulate_prices()
                    # Figure
                    fig = None
                    try:
                        fig = MC.plot_simulation_results(num_of_movements)
                    except Exception:
                        pass
                    if fig is None:
                        fig = plt.gcf()
                    st.pyplot(fig)

                    call_option_price = MC._calculate_call_option_price()
                    put_option_price = MC._calculate_put_option_price()

                    c1, c2 = st.columns(2)
                    c1.metric("Call price", f"{call_option_price:,.2f}")
                    c2.metric("Put price",  f"{put_option_price:,.2f}")
            else:
                st.error("Unable to proceed with calculations due to data fetching error.")
        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")
    else:
        st.info("Click 'Calculate option price' to see results.")


# -------------------- BINOMIAL TREE (inchangé) --------------------
elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
    ticker, strike_price = get_common_inputs()
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10) / 100.0
    st.caption("Theoretical risk-free return (annualized).")
    sigma = st.slider('Sigma (Volatility) (%)', 0, 100, 20) / 100.0
    st.caption("Annualized volatility.")
    exercise_date = st.date_input(
        'Exercise date',
        min_value=datetime.today().date() + timedelta(days=1),
        value=datetime.today().date() + timedelta(days=365)
    )
    st.caption("The date when the option can be exercised.")

    number_of_time_steps = st.slider('Number of time steps', 10, 2000, 500)
    st.caption("The number of periods in the binomial tree. More steps increase accuracy but take longer to compute.")

    if st.button(f'Calculate option price for {ticker}'):
        try:
            with st.spinner('Fetching data...'):
                data = get_stock_data(
                    ticker,
                    start='2020-01-01',
                    end=datetime.today().strftime('%Y-%m-%d')
                )

            if data is not None and not data.empty:
                st.write("Data fetched successfully:")
                st.dataframe(data.tail())

                spot_price = float(data['Close'].iloc[-1])
                days_to_maturity = (exercise_date - datetime.now().date()).days
                if days_to_maturity <= 0:
                    st.error("Exercise date must be in the future.")
                else:
                    BOPM = BinomialTreeModel(
                        underlying_spot_price=spot_price,
                        strike_price=strike_price,
                        days_to_maturity=days_to_maturity,
                        risk_free_rate=risk_free_rate,
                        sigma=sigma,
                        number_of_time_steps=number_of_time_steps
                    )
                    call_option_price = BOPM.CalculateCallOptionPrice()
                    put_option_price = BOPM._calculate_put_option_price()

                    c1, c2 = st.columns(2)
                    c1.metric("Call price", f"{call_option_price:,.2f}")
                    c2.metric("Put price",  f"{put_option_price:,.2f}")
            else:
                st.error("Unable to proceed with calculations due to data fetching error.")
        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")
    else:
        st.info("Click 'Calculate option price' to see results.")
