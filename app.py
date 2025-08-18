import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf

from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel


st.set_page_config(page_title="Option Pricing", layout="wide")
plt.style.use("dark_background")

CALL_BG = "#90EE90"
PUT_BG  = "#FFB6C1"


st.markdown("""
<style>
:root { --bg:#0e1117; --panel:#111827; --text:#e5e7eb; }
.stApp, .block-container { background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background: var(--panel) !important; }
h1,h2,h3,h4,h5,h6,p,span,div,label,small { color: var(--text) !important; }
.stButton > button, .stDownloadButton > button { border-radius:10px !important; font-weight:600 !important; padding:.55rem 1rem !important; }
.kpi-card .kpi-title, .kpi-card .kpi-value { color:#000 !important; }
.kpi-card .kpi-title { font-size:12px; opacity:.85; margin-bottom:4px; }
.kpi-card .kpi-value { font-size:22px; font-weight:800; }
.block-container { padding-top: .25rem !important; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)





class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black Scholes Model'
    MONTE_CARLO   = 'Monte Carlo Simulation'
    BINOMIAL      = 'Binomial Model'


@st.cache_data
def get_stock_data(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    os.makedirs("./CSVs", exist_ok=True)
    path = f'./CSVs/{ticker}_returns.csv'

    def download_and_save():
        df_new = yf.download(ticker, start=start, end=end)
        if not df_new.empty:
            df_new.to_csv(path)
        return df_new

    if force_refresh:
        return download_and_save()

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            last = df.index.max().date()
            today = datetime.today().date()
            if (today - last).days > 2:
                df = download_and_save()
        else:
            df = download_and_save()
    except Exception:
        df = download_and_save()

    return df


def get_common_inputs(default_ticker="AAPL"):
    ticker = st.sidebar.text_input('Ticker symbol', default_ticker)
    st.sidebar.caption("Enter the stock symbol (e.g., AAPL for Apple Inc.)")
    strike_price = st.sidebar.number_input('Strike price', min_value=0.01, max_value=1_000_000.0, value=100.0, step=0.01)
    return ticker, strike_price

def value_card(title: str, value: str, bg: str):
    st.markdown(
        f"""
        <div class="kpi-card" style="
          border-radius:12px; padding:14px 16px; background:{bg};
          text-align:center; border:1px solid rgba(0,0,0,.12);
          box-shadow: inset 0 1px 2px rgba(0,0,0,.25);">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.title('Option Pricing')

pricing_method = st.sidebar.radio(
    'Please select option pricing model',
    options=[m.value for m in OPTION_PRICING_MODEL]
)
st.subheader(f'Pricing method: {pricing_method}')
with st.sidebar:
    st.title(f"📊 {pricing_method}")
    st.markdown(
        '<div style="color:#fff;font-weight:600;margin-bottom:6px;">Created by:</div>',
        unsafe_allow_html=True
    )
    linkedin_url = "https://www.linkedin.com/in/nicolas-morganti/"
    st.markdown(
        f'<a href="{linkedin_url}" target="_blank" '
        'style="text-decoration:none;color:#fff;display:flex;align-items:center;">'
        '<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" '
        'width="20" height="20" style="margin-right:8px;">'
        'Nicolas Morganti</a>',
        unsafe_allow_html=True
    )

# BLACK-SCHOLES

if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
    ticker, strike_price = get_common_inputs()
    risk_free_rate = st.sidebar.slider('Risk-free rate (%)', 0, 100, 10) / 100.0
    sigma = st.sidebar.slider('Sigma (Volatility) (%)', 0, 100, 20) / 100.0
    exercise_date = st.sidebar.date_input('Exercise date', min_value=datetime.today().date() + timedelta(days=1),
                                          value=datetime.today().date() + timedelta(days=365))
    refresh = st.sidebar.checkbox("Force refresh latest prices", value=False)

    # Spot handling
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Spot handling (Black-Scholes)")
    use_market_spot_bs = st.sidebar.toggle("Use market spot", value=True, key="bs_use_market_spot")
    manual_spot_bs = None
    if not use_market_spot_bs:
        manual_spot_bs = st.sidebar.number_input("Manual spot", min_value=0.0, value=100.0, step=0.01, key="bs_manual_spot")

    if st.sidebar.button(f'Calculate option price for {ticker}', type="primary", key="calc_bs"):
        try:
            data = get_stock_data(ticker, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'),
                                  force_refresh=refresh)
            if data is None or data.empty:
                st.error("Unable to fetch data.")
            else:
                spot_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                market_spot = float(data[spot_col].iloc[-1])
                spot_for_pricing = market_spot if use_market_spot_bs else float(manual_spot_bs)

                days_to_maturity = (exercise_date - datetime.now().date()).days
                if days_to_maturity <= 0:
                    st.error("Exercise date must be in the future.")
                else:
                    BSM = BlackScholesModel(spot_for_pricing, strike_price, days_to_maturity, risk_free_rate, sigma)
                    call_price = BSM._calculate_call_option_price()
                    put_price  = BSM._calculate_put_option_price()

                    st.markdown("### Black-Scholes Pricing Model")
                    info_cols = st.columns(5)
                    info_cols[0].metric("Current Asset Price", f"{spot_for_pricing:,.4f}")
                    info_cols[1].metric("Strike Price", f"{strike_price:,.4f}")
                    info_cols[2].metric("Time to Maturity (days)", f"{days_to_maturity}")
                    info_cols[3].metric("Volatility (σ)", f"{sigma:.4f}")
                    info_cols[4].metric("Risk-Free Rate", f"{risk_free_rate:.4f}")

                    k1, k2 = st.columns(2)
                    with k1: value_card("CALL Value", f"${call_price:,.2f}", CALL_BG)
                    with k2: value_card("PUT Value",  f"${put_price:,.2f}",  PUT_BG)

                    st.caption(f"Market spot reference ({spot_col}): ${market_spot:.2f}")

                    # Heatmaps interactives
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### Heatmap Parameters (BS)")
                    s_min = st.sidebar.number_input("Min Spot Price", value=max(1.0, round(spot_for_pricing * 0.8, 2)), key="bs_smin")
                    s_max = st.sidebar.number_input("Max Spot Price", value=round(spot_for_pricing * 1.2, 2), key="bs_smax")
                    v_min = st.sidebar.number_input("Min Volatility (σ)", value=max(0.05, round(sigma * 0.5, 2)), key="bs_vmin")
                    v_max = st.sidebar.number_input("Max Volatility (σ)", value=round(min(1.0, sigma * 1.5), 2), key="bs_vmax")
                    nS = st.sidebar.slider("Spot grid", 10, 60, 21, key="bs_nS")
                    nV = st.sidebar.slider("Vol grid",  10, 60, 21, key="bs_nV")

                    if s_max <= s_min: s_max = s_min + 1e-6
                    if v_max <= v_min: v_max = v_min + 1e-6

                    Ss = np.linspace(s_min, s_max, int(nS))
                    Vs = np.linspace(v_min, v_max, int(nV))
                    call_grid = np.zeros((len(Vs), len(Ss)))
                    put_grid  = np.zeros((len(Vs), len(Ss)))
                    for i, v in enumerate(Vs):
                        for j, s in enumerate(Ss):
                            b = BlackScholesModel(float(s), float(strike_price), int(days_to_maturity),
                                                  float(risk_free_rate), float(v))
                            call_grid[i, j] = b._calculate_call_option_price()
                            put_grid[i, j]  = b._calculate_put_option_price()

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### Call Price Heatmap (BS)")
                        fig1, ax1 = plt.subplots()
                        im1 = ax1.imshow(call_grid, origin="lower", aspect="auto",
                                         extent=[Ss.min(), Ss.max(), Vs.min(), Vs.max()])
                        ax1.set_xlabel("Spot Price"); ax1.set_ylabel("Volatility (σ)")
                        plt.colorbar(im1, ax=ax1)
                        st.pyplot(fig1)
                    with c2:
                        st.markdown("#### Put Price Heatmap (BS)")
                        fig2, ax2 = plt.subplots()
                        im2 = ax2.imshow(put_grid, origin="lower", aspect="auto",
                                         extent=[Ss.min(), Ss.max(), Vs.min(), Vs.max()])
                        ax2.set_xlabel("Spot Price"); ax2.set_ylabel("Volatility (σ)")
                        plt.colorbar(im2, ax=ax2)
                        st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error: {e}")


# MONTE CARLO

elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    ticker, strike_price = get_common_inputs()
    risk_free_rate = st.sidebar.slider('Risk-free rate (%)', 0, 100, 10) / 100.0
    sigma = st.sidebar.slider('Sigma (Volatility) (%)', 0, 100, 20) / 100.0
    exercise_date = st.sidebar.date_input('Exercise date', min_value=datetime.today().date() + timedelta(days=1),
                                          value=datetime.today().date() + timedelta(days=365))
    refresh = st.sidebar.checkbox("Force refresh latest prices (MC)", value=False)
    number_of_simulations = st.sidebar.slider('Number of simulations', 100, 100000, 10000)
    num_of_movements = st.sidebar.slider('Paths to display', 1, 50, 10)


    st.sidebar.markdown("---")
    st.sidebar.markdown("### Spot handling (Monte Carlo)")
    use_market_spot_mc = st.sidebar.toggle("Use market spot", value=True, key="mc_use_market_spot")
    manual_spot_mc = None
    if not use_market_spot_mc:
        manual_spot_mc = st.sidebar.number_input("Manual spot", min_value=0.0, value=100.0, step=0.01, key="mc_manual_spot")

    if st.sidebar.button(f'Calculate option price for {ticker} (MC)', type="primary", key="calc_mc"):
        try:
            data = get_stock_data(ticker, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'),
                                  force_refresh=refresh)
            if data is None or data.empty:
                st.error("Unable to fetch data.")
            else:
                spot_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                market_spot = float(data[spot_col].iloc[-1])
                spot_price = market_spot if use_market_spot_mc else float(manual_spot_mc)

                days_to_maturity = (exercise_date - datetime.now().date()).days
                if days_to_maturity <= 0:
                    st.error("Exercise date must be in the future.")
                else:
                    MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity,
                                           risk_free_rate, sigma, number_of_simulations)
                    MC.simulate_prices()
                    call_price = MC._calculate_call_option_price()
                    put_price  = MC._calculate_put_option_price()

                    st.markdown("### Monte Carlo Pricing Model")
                    info_cols = st.columns(5)
                    info_cols[0].metric("Current Asset Price", f"{spot_price:,.4f}")
                    info_cols[1].metric("Strike Price", f"{strike_price:,.4f}")
                    info_cols[2].metric("Time to Maturity (days)", f"{days_to_maturity}")
                    info_cols[3].metric("Volatility (σ)", f"{sigma:.4f}")
                    info_cols[4].metric("Risk-Free Rate", f"{risk_free_rate:.4f}")

                    k1, k2 = st.columns(2)
                    with k1: value_card("CALL Value", f"${call_price:,.2f}", CALL_BG)
                    with k2: value_card("PUT Value",  f"${put_price:,.2f}",  PUT_BG)


                    T = days_to_maturity / 252
                    n_steps = 100
                    dt = T / n_steps
                    np.random.seed(42)
                    paths = np.zeros((n_steps+1, num_of_movements))
                    paths[0] = spot_price
                    for t in range(1, n_steps+1):
                        z = np.random.standard_normal(num_of_movements)
                        paths[t] = paths[t-1] * np.exp((risk_free_rate - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
                    fig, ax = plt.subplots()
                    ax.plot(paths)
                    ax.set_title("Simulated Price Paths")
                    ax.set_xlabel("Time steps"); ax.set_ylabel("Price level")
                    st.pyplot(fig)

                    st.caption(f"Market spot reference ({spot_col}): ${market_spot:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")


# BINOMIAL

elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
    ticker, strike_price = get_common_inputs()
    risk_free_rate = st.sidebar.slider('Risk-free rate (%)', 0, 100, 10) / 100.0
    sigma = st.sidebar.slider('Sigma (Volatility) (%)', 0, 100, 20) / 100.0
    exercise_date = st.sidebar.date_input('Exercise date', min_value=datetime.today().date() + timedelta(days=1),
                                          value=datetime.today().date() + timedelta(days=365))
    refresh = st.sidebar.checkbox("Force refresh latest prices (Binomial)", value=False)
    number_of_time_steps = st.sidebar.slider('Number of time steps', 10, 2000, 500)

    # Spot handling (Binomial)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Spot handling (Binomial)")
    use_market_spot_bin = st.sidebar.toggle("Use market spot", value=True, key="bin_use_market_spot")
    manual_spot_bin = None
    if not use_market_spot_bin:
        manual_spot_bin = st.sidebar.number_input("Manual spot", min_value=0.0, value=100.0, step=0.01, key="bin_manual_spot")

    if st.sidebar.button(f'Calculate option price for {ticker} (Binomial)', type="primary", key="calc_bin"):
        try:
            data = get_stock_data(ticker, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'),
                                  force_refresh=refresh)
            if data is None or data.empty:
                st.error("Unable to fetch data.")
            else:
                spot_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                market_spot = float(data[spot_col].iloc[-1])
                spot_price = market_spot if use_market_spot_bin else float(manual_spot_bin)

                days_to_maturity = (exercise_date - datetime.now().date()).days
                if days_to_maturity <= 0:
                    st.error("Exercise date must be in the future.")
                else:
                    BOPM = BinomialTreeModel(spot_price, strike_price, days_to_maturity,
                                             risk_free_rate, sigma, number_of_time_steps)
                    call_price = BOPM.CalculateCallOptionPrice()
                    put_price  = BOPM._calculate_put_option_price()

                    st.markdown("### Binomial Tree Pricing Model")
                    info_cols = st.columns(5)
                    info_cols[0].metric("Current Asset Price", f"{spot_price:,.4f}")
                    info_cols[1].metric("Strike Price", f"{strike_price:,.4f}")
                    info_cols[2].metric("Time to Maturity (days)", f"{days_to_maturity}")
                    info_cols[3].metric("Volatility (σ)", f"{sigma:.4f}")
                    info_cols[4].metric("Risk-Free Rate", f"{risk_free_rate:.4f}")

                    k1, k2 = st.columns(2)
                    with k1: value_card("CALL Value", f"${call_price:,.2f}", CALL_BG)
                    with k2: value_card("PUT Value",  f"${put_price:,.2f}",  PUT_BG)

                    # Heatmaps interactives (Binomial)
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### Heatmap Parameters (Binomial)")
                    s_min = st.sidebar.number_input("Min Spot Price", value=max(1.0, round(spot_price * 0.8, 2)), key="bin_smin")
                    s_max = st.sidebar.number_input("Max Spot Price", value=round(spot_price * 1.2, 2), key="bin_smax")
                    v_min = st.sidebar.number_input("Min Volatility (σ)", value=max(0.05, round(sigma * 0.5, 2)), key="bin_vmin")
                    v_max = st.sidebar.number_input("Max Volatility (σ)", value=round(min(1.0, sigma * 1.5), 2), key="bin_vmax")
                    nS = st.sidebar.slider("Spot grid", 10, 60, 21, key="bin_nS")
                    nV = st.sidebar.slider("Vol grid",  10, 60, 21, key="bin_nV")

                    if s_max <= s_min: s_max = s_min + 1e-6
                    if v_max <= v_min: v_max = v_min + 1e-6

                    Ss = np.linspace(s_min, s_max, int(nS))
                    Vs = np.linspace(v_min, v_max, int(nV))
                    call_grid = np.zeros((len(Vs), len(Ss)))
                    put_grid  = np.zeros((len(Vs), len(Ss)))
                    for i, v in enumerate(Vs):
                        for j, s in enumerate(Ss):
                            b = BinomialTreeModel(float(s), float(strike_price), int(days_to_maturity),
                                                  float(risk_free_rate), float(v), number_of_time_steps)
                            call_grid[i, j] = b.CalculateCallOptionPrice()
                            put_grid[i, j]  = b._calculate_put_option_price()

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### Call Price Heatmap (Binomial)")
                        fig1, ax1 = plt.subplots()
                        im1 = ax1.imshow(call_grid, origin="lower", aspect="auto",
                                         extent=[Ss.min(), Ss.max(), Vs.min(), Vs.max()])
                        ax1.set_xlabel("Spot Price"); ax1.set_ylabel("Volatility (σ)")
                        plt.colorbar(im1, ax=ax1)
                        st.pyplot(fig1)
                    with c2:
                        st.markdown("#### Put Price Heatmap (Binomial)")
                        fig2, ax2 = plt.subplots()
                        im2 = ax2.imshow(put_grid, origin="lower", aspect="auto",
                                         extent=[Ss.min(), Ss.max(), Vs.min(), Vs.max()])
                        ax2.set_xlabel("Spot Price"); ax2.set_ylabel("Volatility (σ)")
                        plt.colorbar(im2, ax=ax2)
                        st.pyplot(fig2)

                    st.caption(f"Market spot reference ({spot_col}): ${market_spot:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
