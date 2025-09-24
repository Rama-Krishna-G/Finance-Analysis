import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_stock_data(start_date='2024-01-01', end_date='2025-08-31', initial_price=100.0):
    """
    Generate synthetic stock price data with realistic patterns.
    
    Parameters:
    - start_date: Start date in 'YYYY-MM-DD' format
    - end_date: End date in 'YYYY-MM-DD' format
    - initial_price: Starting price of the stock
    
    Returns:
    - DataFrame with synthetic stock data
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    n_days = len(dates)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base trend (slight upward trend)
    trend = np.linspace(0, 0.5, n_days)  # 50% increase over the period
    
    # Generate seasonality (weekly and monthly patterns)
    day_of_week = np.array([d.weekday() for d in dates])
    month = np.array([d.month for d in dates])
    
    # Weekly pattern (lower volatility on Mondays and Fridays)
    weekly_pattern = 0.02 * np.sin(2 * np.pi * np.arange(n_days) / 5)
    
    # Monthly pattern (end of month effect)
    day_of_month = np.array([d.day for d in dates])
    monthly_pattern = 0.01 * np.sin(2 * np.pi * (day_of_month - 1) / 30)
    
    # Random noise
    noise = 0.01 * np.random.randn(n_days)
    
    # Combine all components
    log_returns = 0.0005 + 0.01 * (trend + weekly_pattern + monthly_pattern + noise)
    
    # Calculate price series
    prices = initial_price * np.exp(np.cumsum(log_returns))
    
    # Generate OHLC data with some random variation
    data = []
    for i in range(n_days):
        if i == 0:
            prev_close = initial_price
        else:
            prev_close = data[-1]['Close']
            
        close = prices[i]
        high = close * (1 + 0.01 * np.abs(np.random.randn() * 0.5))
        low = close * (1 - 0.01 * np.abs(np.random.randn() * 0.5))
        open_price = prev_close * (1 + 0.005 * np.random.randn())
        
        # Ensure high > low and proper ordering
        high = max(open_price, close, high)
        low = min(open_price, close, low)
        
        # Generate volume (in thousands)
        volume = int(10000 + 5000 * np.random.rand())
        
        data.append({
            'Date': dates[i].strftime('%b %d, %Y'),
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Adj Close': round(close, 2),  # Same as close for simplicity
            'Volume': volume * 1000  # Convert to actual volume
        })
    
    return pd.DataFrame(data)

def save_to_excel(df, filename='synthetic_stock_data_2024_2025.xlsx'):
    """Save the generated data to an Excel file."""
    df.to_excel(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Generate data
    print("Generating synthetic stock data from Jan 2024 to Aug 2025...")
    df = generate_stock_data()
    
    # Save to Excel
    output_file = "data/raw/synthetic_stock_data_2024_2025.xlsx"
    save_to_excel(df, output_file)
    
    # Display sample of the generated data
    print("\nSample of the generated data:")
    print(df.head())
    print("\nData generation complete!")
