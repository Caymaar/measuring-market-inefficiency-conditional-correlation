import yfinance as yf
import os
import sys
sys.path.append('.')
from src.utils import get_config

def main():
    config = get_config()
    data_folder = config['paths']['data_folder']
    
    # Create data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    # Get stocks from config
    stocks = dict(config['stocks'])

    for name, symbol in stocks.items():
        print(f"Downloading data for {name} ({symbol})...")
        ticker = yf.Ticker(symbol)
        # Get full historical data (max available)
        df = ticker.history(period="max")

        df = df[['Close']]

        # Rename the column to 'Close'
        df.rename(columns={'Close': name}, inplace=True)

        # Save to CSV file with the ticker name
        file_name = os.path.join(data_folder, f"{name}.csv")
        df.to_csv(file_name)
        print(f"Data saved to {file_name}")

if __name__ == "__main__":
    main()