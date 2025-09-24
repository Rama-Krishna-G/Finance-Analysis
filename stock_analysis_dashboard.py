import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import logging

# Configure logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('fbprophet').setLevel(logging.WARNING)

# Set page config
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Dark theme with black background
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .stApp {
        background-color: #000000;
    }
    
    /* Text color */
    .stText, .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1v3fvcr {
        background-color: #121212 !important;
        color: #ffffff !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox>div>div, .stTextInput>div>div>input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border-color: #333333 !important;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #4CAF50 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: 1px solid #45a049 !important;
    }
    
    /* Plotly chart styling */
    .js-plotly-plot, .plotly {
        background-color: #121212 !important;
    }
    
    /* Tables and dataframes */
    .stDataFrame, .dataframe {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    
    /* Hover effects */
    .stDataFrame:hover, .element-container:hover {
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Available stocks with their display names and Yahoo Finance symbols
STOCKS = {
    'Infosys (NSE)': 'INFY.NS',
    'TCS (NSE)': 'TCS.NS',
    'HDFC Bank (NSE)': 'HDFCBANK.NS',
    'Reliance (NSE)': 'RELIANCE.NS',
    'ICICI Bank (NSE)': 'ICICIBANK.NS',
    'Infosys (BSE)': '500209.BO',
    'TCS (BSE)': '532540.BO',
    'HDFC Bank (BSE)': '500180.BO',
    'Reliance (BSE)': '500325.BO',
    'ICICI Bank (BSE)': '532174.BO'
}

def fetch_stock_data(symbol, period='1y'):
    """Fetch historical stock data with appropriate interval based on period"""
    stock = yf.Ticker(symbol)
    
    # First, try to get info to check if the symbol is valid
    try:
        info = stock.info
        if not info:
            st.error(f"No information found for symbol: {symbol}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching information for {symbol}: {str(e)}")
        return pd.DataFrame()
    
    # Map period to appropriate interval
    period_map = {
        '1d': ('1d', '1m'),    # 1 day with 1-minute intervals
        '3d': ('3d', '5m'),    # 3 days with 5-minute intervals
        '5d': ('5d', '15m'),   # 5 days with 15-minute intervals
        '1mo': ('1mo', '1d'),  # 1 month with daily data
        '3mo': ('3mo', '1d'),  # 3 months with daily data
        '6mo': ('6mo', '1d'),  # 6 months with daily data
        '1y': ('1y', '1d'),    # 1 year with daily data
        '2y': ('2y', '1d'),    # 2 years with daily data
        '5y': ('5y', '1wk'),   # 5 years with weekly data
        '10y': ('10y', '1mo'), # 10 years with monthly data
        'max': ('max', '3mo')  # Maximum available data with quarterly intervals
    }
    
    period_str, interval = period_map.get(period, ('1y', '1d'))
    
    try:
        # Try with the specified interval first
        try:
            hist = stock.history(period=period_str, interval=interval)
            # If we get data but it's too old, try with a different approach
            if not hist.empty and (datetime.now() - hist.index[-1]).days > 7:
                hist = stock.history(period=period_str, interval='1d')
        except Exception as e:
            # If that fails, try with default interval
            hist = stock.history(period=period_str)
            
        # If still no data, try with a larger period
        if hist.empty and period_str in ['1mo', '3mo']:
            hist = stock.history(period='6mo', interval='1d')
            if not hist.empty:
                # Filter to the requested period
                end_date = hist.index[-1]
                if period_str == '1mo':
                    start_date = end_date - pd.DateOffset(months=1)
                else:  # 3mo
                    start_date = end_date - pd.DateOffset(months=3)
                hist = hist[hist.index >= start_date]
            
        # If still no data, try with a different period format (e.g., '1m' vs '1mo')
        if hist.empty and period in ['1m', '3m', '6m']:
            alt_period = period + 'o'  # Try '1mo' instead of '1m'
            hist = stock.history(period=alt_period, interval='1d')
        
        # If we have some data but it's too old, try to get more recent data
        if not hist.empty and len(hist) < 5:
            # First, ensure we have timezone-naive timestamps for comparison
            hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
            
            # Get the most recent data with daily intervals
            recent_data = stock.history(period='1y', interval='1d')
            if not recent_data.empty:
                recent_data.index = recent_data.index.tz_localize(None) if recent_data.index.tz is not None else recent_data.index
                
                # Calculate date range for the requested period
                end_date = pd.Timestamp.now()
                if period == '1m':
                    start_date = end_date - pd.DateOffset(months=1)
                elif period == '3m':
                    start_date = end_date - pd.DateOffset(months=3)
                elif period == '6m':
                    start_date = end_date - pd.DateOffset(months=6)
                else:
                    start_date = end_date - pd.DateOffset(years=1)
                
                # Filter the recent data to the requested period
                mask = (recent_data.index >= start_date) & (recent_data.index <= end_date)
                filtered_data = recent_data.loc[mask]
                
                # Use the filtered data if we have any
                if not filtered_data.empty:
                    hist = filtered_data
        
        # Ensure we have timezone-naive index before returning
        if not hist.empty and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
            
        return hist
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol} (period: {period}): {str(e)}")
        # Try one last time with default parameters
        try:
            hist = stock.history(period=period)
            if not hist.empty and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            return hist
        except Exception as e2:
            st.error(f"Final attempt failed: {str(e2)}")
            return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def predict_future_prices(df, days=30):
    """Predict future prices using moving average crossover strategy"""
    try:
        if len(df) < 20:  # Minimum 20 days of data
            return None
            
        prices = df['Close']
        
        # Calculate short and long term moving averages
        short_window = 5
        long_window = 20
        
        # Calculate moving averages
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        
        # Determine the trend based on MA crossover
        current_trend = 'up' if short_ma.iloc[-1] > long_ma.iloc[-1] else 'down'
        
        # Calculate recent volatility (standard deviation of returns)
        returns = prices.pct_change().dropna()
        volatility = returns.std() if len(returns) > 5 else 0.01
        
        # Generate future dates (business days only)
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='B'
        )
        
        # Initialize forecast with last price
        forecast_prices = [prices.iloc[-1]]
        
        # Generate forecast based on trend
        for i in range(1, days):
            # Base change depends on the trend
            if current_trend == 'up':
                # In uptrend, slightly positive bias
                base_change = np.random.normal(0.0005, volatility)
            else:
                # In downtrend, slightly negative bias
                base_change = np.random.normal(-0.0003, volatility)
            
            # Add some randomness
            random_component = np.random.normal(0, volatility * 0.5)
            
            # Calculate next price
            next_price = forecast_prices[-1] * (1 + base_change + random_component)
            
            # Ensure price doesn't change too much in one day (max 2%)
            max_daily_change = 0.02
            next_price = max(
                forecast_prices[-1] * (1 - max_daily_change),
                min(next_price, forecast_prices[-1] * (1 + max_daily_change))
            )
            
            forecast_prices.append(next_price)
        
        # Ensure we have the same number of dates and prices
        num_days = min(len(future_dates), len(forecast_prices[1:]))
        
        # Create a series with the forecasted values
        forecast_series = pd.Series(
            data=forecast_prices[1:num_days+1],  # Skip the first element (last actual price)
            index=future_dates[:num_days],
            name='Forecast'
        )
        
        # Add some visualization of the prediction confidence
        st.markdown("""
        **Prediction Method:** Moving Average Crossover (5/20 day)
        - **Current Trend:** {}
        - **Recent Volatility:** {:.2%} (daily)
        """.format(current_trend.upper(), volatility))
        
        return forecast_series
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def plot_stock_data(df, forecast=None):
    """Create interactive stock price chart"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#2ecc71',
        decreasing_line_color='#e74c3c'
    ))
    
    # Add moving averages if they exist
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='#3498db', width=2)
        ))
    
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_200'],
            name='SMA 200',
            line=dict(color='#9b59b6', width=2)
        ))
    
    # Add forecast if available
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast,
            name='Forecast',
            line=dict(color='#f39c12', width=2, dash='dot')
        ))
    
    # Update layout
    fig.update_layout(
        title='Stock Price Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        height=600
    )
    
    return fig

def plot_rsi(df):
    """Create RSI chart"""
    if 'RSI' not in df.columns:
        return None
        
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                           line=dict(color='#3498db', width=2), 
                           name='RSI'))
    
    # Overbought and oversold levels
    fig.add_hline(y=70, line_dash='dash', line_color='#e74c3c', 
                 annotation_text='Overbought (70)', annotation_position='top right')
    fig.add_hline(y=30, line_dash='dash', line_color='#2ecc71',
                 annotation_text='Oversold (30)', annotation_position='bottom right')
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig

def display_stock_metrics(df, forecast=None):
    """Display key stock metrics with proper error handling"""
    if df is None or df.empty:
        st.warning("No data available to display metrics.")
        return
    
    try:
        # Ensure we have the necessary columns
        if 'Close' not in df.columns:
            st.warning("Missing price data in the dataset.")
            return
            
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        # Calculate daily change with error handling
        try:
            price_change = latest['Close'] - previous['Close']
            percent_change = (price_change / previous['Close']) * 100 if previous['Close'] != 0 else 0
        except (KeyError, IndexError, TypeError) as e:
            st.warning(f"Error calculating price change: {str(e)}")
            price_change = 0
            percent_change = 0
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Current Price
        with col1:
            try:
                st.metric("Current Price", f"₹{latest['Close']:,.2f}" if 'Close' in latest else "N/A")
            except Exception as e:
                st.metric("Current Price", "N/A")
        
        # Daily Change
        with col2:
            try:
                delta_color = "inverse" if price_change < 0 else "normal"
                st.metric("Daily Change", 
                         f"₹{price_change:,.2f}", 
                         f"{percent_change:+.2f}%" if percent_change is not None else "N/A",
                         delta_color=delta_color)
            except Exception as e:
                st.metric("Daily Change", "N/A")
        
        # 30-Day Forecast
        with col3:
            try:
                if forecast is not None and len(forecast) > 0:
                    pred_change = forecast[-1] - latest['Close']
                    pred_percent = (pred_change / latest['Close']) * 100 if latest['Close'] != 0 else 0
                    delta_color = "inverse" if pred_change < 0 else "normal"
                    st.metric("30-Day Forecast", 
                             f"₹{forecast[-1]:,.2f}",
                             f"{pred_percent:+.2f}%",
                             delta_color=delta_color)
                else:
                    st.metric("30-Day Forecast", "N/A")
            except Exception as e:
                st.metric("30-Day Forecast", "N/A")
        
        # Volume
        with col4:
            try:
                if 'Volume' in latest and pd.notna(latest['Volume']):
                    st.metric("Volume", f"{int(latest['Volume']):,}")
                else:
                    st.metric("Volume", "N/A")
            except Exception as e:
                st.metric("Volume", "N/A")
                
    except Exception as e:
        st.error(f"An error occurred while displaying metrics: {str(e)}")

def main():
    st.title("📈 Indian Stock Analysis & Prediction Dashboard")
    st.markdown("Analyze historical performance and get predictions for Indian stocks")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Stock selection
    selected_stock_name = st.sidebar.selectbox(
        "Select a stock:", 
        list(STOCKS.keys()),
        index=0
    )
    
    # Date range selection with valid yfinance periods
    period_options = [
        "1mo",    # 1 month
        "3mo",    # 3 months
        "6mo",    # 6 months
        "1y",     # 1 year
        "2y",     # 2 years
        "5y",     # 5 years
        "10y",    # 10 years
        "max"     # Maximum available
    ]
    
    period_display_names = {
        "1mo": "1 Month",
        "3mo": "3 Months",
        "6mo": "6 Months",
        "1y": "1 Year",
        "2y": "2 Years",
        "5y": "5 Years",
        "10y": "10 Years",
        "max": "Max Available"
    }
    
    selected_period = st.sidebar.selectbox(
        "Select time period:",
        options=period_options,
        format_func=lambda x: period_display_names[x],
        index=3  # Default to 1 year
    )
    period = selected_period
    
    # Get stock symbol
    symbol = STOCKS[selected_stock_name]
    
    # Add prediction days slider
    prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=30,
        max_value=180,
        value=30,
        step=1,
        help="Select number of days to predict (30-180)"
    )
    
    # Fetch and process data
    with st.spinner('Fetching stock data...'):
        try:
            df = fetch_stock_data(symbol, period)
            
            if df is None or df.empty:
                st.error("❌ No data available for the selected stock and period. Please try a different stock or time period.")
                st.info("💡 Try selecting a longer time period or a different stock.")
                return
                
            st.success(f"✅ Successfully fetched {len(df)} data points")
            
            # Calculate technical indicators with available data
            try:
                df = calculate_technical_indicators(df)
                if len(df) < 50:
                    st.warning("⚠️ Limited data points. Some indicators may be less accurate.")
            except Exception as e:
                # Fallback to basic indicators if full calculation fails
                try:
                    df['SMA_20'] = df['Close'].rolling(window=min(20, len(df)), min_periods=1).mean()
                    if 'Close' in df and len(df) >= 14:  # Minimum for RSI
                        delta = df['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        df['RSI'] = 100 - (100 / (1 + rs))
                    st.warning("⚠️ Using basic indicators due to limited data.")
                except Exception as e:
                    st.warning(f"⚠️ Could not calculate technical indicators: {str(e)}")
            
            # Make prediction if requested
            forecast = None  # Initialize forecast variable
            if st.sidebar.checkbox("Show Price Prediction", value=False):
                with st.spinner("Generating price prediction..."):
                    try:
                        # Adjust prediction days based on available data
                        available_days = len(df)
                        if available_days < 10:
                            st.warning("⚠️ Very limited data available. Predictions may be less accurate.")
                        
                        # Ensure we're not trying to predict more days than we have data for
                        adjusted_prediction_days = min(prediction_days, available_days)
                        
                        forecast = predict_future_prices(df, days=adjusted_prediction_days)
                        if forecast is not None and not forecast.empty and len(forecast) > 0:
                            st.success(f"✅ Generated {len(forecast)}-day price forecast")
                        else:
                            st.warning("⚠️ Could not generate price prediction. Not enough data.")
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate price prediction: {str(e)}")
                        forecast = None
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.error("Please try a different stock or time period.")
            return
    
    # Display metrics
    display_stock_metrics(df, forecast)
    
    # Display charts
    st.plotly_chart(plot_stock_data(df, forecast), use_container_width=True)
    
    # Show RSI chart if data is available
    rsi_chart = plot_rsi(df)
    if rsi_chart:
        st.plotly_chart(rsi_chart, use_container_width=True)
    
    # Show raw data
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(10))
    
    # Show company info
    try:
        with st.expander("Company Information"):
            stock = yf.Ticker(symbol)
            info = stock.info
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Company Profile")
                st.write(f"**Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
                
            with col2:
                st.subheader("Key Metrics")
                st.write(f"**Market Cap:** ₹{info.get('marketCap', 'N/A'):,}" if 'marketCap' in info else "**Market Cap:** N/A")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}" if 'trailingPE' in info else "**P/E Ratio:** N/A")
                st.write(f"**52-Week High:** ₹{info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}" if 'fiftyTwoWeekHigh' in info else "**52-Week High:** N/A")
                st.write(f"**52-Week Low:** ₹{info.get('fiftyTwoWeekLow', 'N/A'):,.2f}" if 'fiftyTwoWeekLow' in info else "**52-Week Low:** N/A")
    
    except Exception as e:
        st.warning(f"Could not load company information: {str(e)}")

if __name__ == "__main__":
    main()
