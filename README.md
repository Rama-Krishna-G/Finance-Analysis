# 📈 Indian Stock Analysis & Prediction Dashboard

A powerful and interactive dashboard for analyzing Indian stock market data with price prediction capabilities. Built with Python, Streamlit, and Plotly.

![Dashboard Screenshot](https://via.placeholder.com/800x500/121212/4CAF50?text=Stock+Analysis+Dashboard)

## ✨ Features

- **Real-time Data**: Fetch live stock data for major Indian companies
- **Technical Analysis**: View candlestick charts with moving averages
- **Price Prediction**: Get 30-180 day price forecasts
- **RSI Indicator**: Monitor overbought/oversold conditions
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Theme**: Easy on the eyes with a modern dark interface

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-analysis-dashboard.git
   cd stock-analysis-dashboard
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

```bash
streamlit run stock_analysis_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## 🛠️ Dependencies

- streamlit
- pandas
- numpy
- yfinance
- plotly
- prophet
- python-dotenv

## 📊 Available Stocks

- Infosys (NSE)
- TCS (NSE)
- HDFC Bank (NSE)
- Reliance (NSE)
- HDFC (NSE)
- ICICI Bank (NSE)
- SBI (NSE)
- Wipro (NSE)
- Tech Mahindra (NSE)
- Bajaj Finance (NSE)
- Asian Paints (NSE)
- HUL (NSE)
- ITC (NSE)
- Kotak Bank (NSE)
- Axis Bank (NSE)

## 📈 Features in Detail

### 1. Stock Selection
Choose from a list of major Indian companies listed on NSE.

### 2. Time Period Selection
Analyze data from different time periods:
- 1 Month
- 3 Months
- 6 Months
- 1 Year
- 2 Years
- 5 Years
- 10 Years
- Max Available

### 3. Technical Indicators
- 50-day and 200-day Simple Moving Averages (SMA)
- Relative Strength Index (RSI)
- Volume Analysis

### 4. Price Prediction
Get AI-powered price predictions for the next 30-180 days using Facebook's Prophet model.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or feedback, please open an issue or contact the maintainers.

---

Made with ❤️ by [Your Name] | [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fyourusername%2Fstock-analysis-dashboard)](https://twitter.com/yourhandle)
