import pandas as pd

def load_data(file_path):
    """
    Load and preprocess the financial data.
    
    Args:
        file_path: Path to the Excel file containing the data
        
    Returns:
        DataFrame with processed data
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Create a mapping of lowercase column names to original names
    col_mapping = {col.lower(): col for col in df.columns}
    
    # Define required columns (case-insensitive)
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # Check if all required columns exist (case-insensitive)
    missing_columns = [col for col in required_columns if col not in col_mapping]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Ensure numeric columns are numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate returns
    df['return'] = df['close'].pct_change()
    
    # Reset index to ensure it's clean
    df = df.reset_index(drop=True)
    
    return df

def calculate_metrics(df):
    """Calculate key financial metrics."""
    try:
        # Debug: Print available columns
        print("\n[DEBUG] DataFrame columns:", df.columns.tolist())
        
        # Check if we have a date column or if date is in index
        if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df['date'] = df.index
        
        # Ensure we have the required columns
        required_columns = ['date', 'close', 'return']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        # Calculate metrics
        metrics = {
            'start_date': df['date'].min().strftime('%Y-%m-%d'),
            'end_date': df['date'].max().strftime('%Y-%m-%d'),
            'total_days': len(df),
            'initial_price': df['close'].iloc[0],
            'final_price': df['close'].iloc[-1],
            'total_return': (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100,
            'avg_daily_return': df['return'].mean() * 100,
            'volatility': df['return'].std() * 100 if len(df) > 1 else 0,
            'max_drawdown': ((df['close'] / df['close'].cummax() - 1).min()) * 100
        }
        
        # Debug: Print calculated metrics
        print("[DEBUG] Calculated metrics:", metrics)
        
        return metrics
        
    except Exception as e:
        # Debug: Print error details
        print(f"[ERROR] Error in calculate_metrics: {str(e)}")
        print("DataFrame info:")
        print(df.info())
        print("\nDataFrame head:")
        print(df.head())
        
        # Return default metrics with error information
        return {
            'error': str(e),
            'available_columns': df.columns.tolist(),
            'index_type': str(type(df.index))
        }
