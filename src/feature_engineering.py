import pandas as pd
import numpy as np
import glob
import os

def process_tick_data(data_dir: str = "data", symbol: str = "BTCUSDT", freq: str = "1min") -> pd.DataFrame:
    """
    Loads raw tick data from Parquet files, resamples it to a fixed time grid,
    and engineers microstructural alpha features.
    """
    # 1. Load all Parquet files for the specified symbol
    file_pattern = os.path.join(data_dir, f"{symbol}_aggTrades_*.parquet")
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}. Run ingestion first.")
        
    print(f"Loading {len(files)} days of data...")
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list)

    # --- THE BULLETPROOF TIMESTAMP RECONSTRUCTION ---
    print("Validating and aligning timestamps...")
    if 'timestamp' in df.columns:
        # If the median timestamp has more than 14 digits, it is in microseconds.
        if df['timestamp'].median() > 1e14:
            df.index = pd.to_datetime(df['timestamp'], unit='us')
        else:
            df.index = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        raise ValueError("Raw 'timestamp' column missing. Check ingestion script.")
    
    # Ensure chronological order after index recreation
    df = df.sort_index()
    # ------------------------------------------------

    print("Resampling tick data and engineering features...")
    
    # 2. Separate Buy and Sell Volumes based on the maker flag
    df['buy_vol'] = np.where(~df['is_buyer_maker'], df['quantity'], 0.0)
    df['sell_vol'] = np.where(df['is_buyer_maker'], df['quantity'], 0.0)

    # 3. Resample to a uniform time grid (e.g., 1-minute bars)
    resampled = pd.DataFrame()
    resampled['close_price'] = df['price'].resample(freq).last()
    resampled['volume'] = df['quantity'].resample(freq).sum()
    resampled['buy_vol'] = df['buy_vol'].resample(freq).sum()
    resampled['sell_vol'] = df['sell_vol'].resample(freq).sum()
    
    # Forward fill missing prices for intervals with zero trades, set volume to 0
    resampled['close_price'] = resampled['close_price'].ffill()
    resampled.fillna({'volume': 0, 'buy_vol': 0, 'sell_vol': 0}, inplace=True)

    # 4. Calculate Order Book Imbalance (OBI)
    # Adding a small epsilon (1e-8) to the denominator prevents DivisionByZero errors
    resampled['obi'] = (resampled['buy_vol'] - resampled['sell_vol']) / (resampled['volume'] + 1e-8)
    
    # 5. Calculate volatility (Rolling standard deviation of log returns)
    resampled['log_return'] = np.log(resampled['close_price'] / resampled['close_price'].shift(1))
    resampled['volatility_20'] = resampled['log_return'].rolling(window=20).std()

    # 6. Create the Target Variable (Label): 5-period forward log return
    horizon = 5
    resampled['target_fwd_return'] = np.log(resampled['close_price'].shift(-horizon) / resampled['close_price'])

    # Drop NaN values created by rolling windows and shifting
    resampled.dropna(inplace=True)
    
    print(f"Feature engineering complete. Final matrix shape: {resampled.shape}")
    return resampled

if __name__ == "__main__":
    # Test the pipeline
    features_df = process_tick_data()
    print(features_df[['close_price', 'obi', 'target_fwd_return']].head())
    
    # Save the feature matrix for the ML model
    features_df.to_parquet("data/feature_matrix.parquet", engine="pyarrow")