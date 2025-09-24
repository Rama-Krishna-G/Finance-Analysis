import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def prepare_data_for_prophet(df, target_col='close'):
    """
    Prepare data for Prophet model.
    
    Args:
        df: DataFrame with datetime index and target column
        target_col: Name of the target column to predict
        
    Returns:
        DataFrame with 'ds' (date) and 'y' (target) columns
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Check if we have a date column or index
    if 'date' in df.columns:
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Set date as index if not already
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.set_index('date')
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame must have a datetime index or 'date' column")
    
    # Create Prophet-compatible dataframe
    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y': df[target_col]
    })
    
    return prophet_df

def train_prophet_model(df, target_col='close', **prophet_kwargs):
    """
    Train a Prophet model on the given data.
    
    Args:
        df: DataFrame with the data
        target_col: Name of the target column
        **prophet_kwargs: Additional arguments to pass to Prophet
        
    Returns:
        tuple: (trained_model, performance_metrics)
    """
    # Prepare data
    prophet_df = prepare_data_for_prophet(df, target_col)
    
    # Initialize and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        **prophet_kwargs
    )
    
    model.fit(prophet_df)
    
    # Perform cross-validation
    df_cv = cross_validation(
        model,
        initial='365 days',
        period='30 days',
        horizon='90 days'
    )
    
    # Calculate performance metrics
    df_p = performance_metrics(df_cv)
    
    return model, df_p

def make_future_dataframe(model, periods, freq='D', include_history=True):
    """
    Make a future dataframe for forecasting.
    
    Args:
        model: Fitted Prophet model
        periods: Number of periods to forecast
        freq: Frequency of the data ('D' for daily, 'B' for business days, etc.)
        include_history: Whether to include historical data in the forecast
        
    Returns:
        DataFrame with future dates
    """
    return model.make_future_dataframe(
        periods=periods,
        freq=freq,
        include_history=include_history
    )

def predict_future(model, future_df):
    """
    Make predictions using a trained Prophet model.
    
    Args:
        model: Trained Prophet model
        future_df: DataFrame with future dates (from make_future_dataframe)
        
    Returns:
        DataFrame with predictions and uncertainty intervals
    """
    forecast = model.predict(future_df)
    return forecast

def get_prediction_components(forecast):
    """
    Extract prediction components from a Prophet forecast.
    
    Args:
        forecast: DataFrame returned by predict_future
        
    Returns:
        dict: Dictionary with trend, yearly, and weekly components
    """
    components = {
        'trend': forecast[['ds', 'trend']].copy(),
        'yearly': forecast[['ds', 'yearly']].copy(),
        'weekly': forecast[['ds', 'weekly']].copy()
    }
    
    if 'additive_terms' in forecast.columns:
        components['additive'] = forecast[['ds', 'additive_terms']].copy()
    
    return components

def plot_forecast(model, forecast, uncertainty=True, plot_components=True):
    """
    Plot the forecast from a Prophet model.
    
    Args:
        model: Trained Prophet model
        forecast: DataFrame with predictions
        uncertainty: Whether to show uncertainty intervals
        plot_components: Whether to plot components
        
    Returns:
        matplotlib Figure object
    """
    fig1 = model.plot(forecast, uncertainty=uncertainty)
    
    if plot_components:
        fig2 = model.plot_components(forecast)
        return fig1, fig2
    
    return fig1

def predict_stock_prices(df, target_col='close', periods=90, freq='B', **prophet_kwargs):
    """
    End-to-end function to predict future stock prices.
    
    Args:
        df: DataFrame with historical data. Must contain 'ds' (date) and 'y' (target) columns.
        target_col: Name of the target column (only used if 'y' column doesn't exist)
        periods: Number of periods to forecast
        freq: Frequency of the data
        **prophet_kwargs: Additional arguments for Prophet
        
    Returns:
        tuple: (forecast, model, performance_metrics)
    """
    import warnings
    import logging
    from prophet import Prophet
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Disable Prophet's internal logging
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').propagate = False
    
    # Suppress all cmdstanpy messages
    import sys
    import os
    
    # Redirect stdout and stderr to devnull
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Check if we have the required columns
    if 'ds' not in df.columns or 'y' not in df.columns:
        # Try to use the first column as date if not specified
        if 'date' in df.columns and 'ds' not in df.columns:
            df = df.rename(columns={'date': 'ds'})
        
        # Try to use target_col as y if not specified
        if target_col in df.columns and 'y' not in df.columns:
            df = df.rename(columns={target_col: 'y'})
    
    # Verify we have the required columns
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("DataFrame must contain 'ds' (date) and 'y' (target) columns, or 'date' and '{}' columns".format(target_col))
    
    # Ensure proper data types
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Drop any rows with missing values
    df = df.dropna(subset=['ds', 'y'])
    
    # Sort by date
    df = df.sort_values('ds')
    
    # Check if we have enough data
    if len(df) < 30:
        raise ValueError(f"Not enough data points for prediction. Need at least 30, got {len(df)}")
    
    # Set default parameters
    default_params = {
        'yearly_seasonality': 'auto',  # Let Prophet decide based on data
        'weekly_seasonality': 'auto',  # Let Prophet decide based on data
        'daily_seasonality': False,    # Disable daily seasonality for daily data
        'seasonality_mode': 'additive',
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'mcmc_samples': 0,
        'uncertainty_samples': 1000,
    }
    
    # Update with any user-provided parameters
    default_params.update(prophet_kwargs)
    
    # Initialize and fit the model
    model = Prophet(**default_params)
    
    # Fit the model with a progress bar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq=freq, include_history=True)
    
    # Make predictions
    forecast = model.predict(future)
    
    # Prepare empty performance metrics
    performance = pd.DataFrame({
        'mape': [float('nan')],
        'rmse': [float('nan')],
        'mse': [float('nan')],
        'mdape': [float('nan')],
        'coverage': [float('nan')]
    })
    
    # Restore stdout and stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    return forecast, model, performance
