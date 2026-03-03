import asyncio
import aiohttp
import pandas as pd
import zipfile
import io
import os
from datetime import datetime, timedelta

# Constants for the Binance Public Data Archive
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", 
           "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]

async def download_and_process(session: aiohttp.ClientSession, symbol: str, date_str: str, output_dir: str, semaphore: asyncio.Semaphore):
    """
    Asynchronously downloads, extracts, and converts Binance tick data to Parquet.
    """
    file_name = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{file_name}"
    parquet_filename = os.path.join(output_dir, f"{symbol}_aggTrades_{date_str}.parquet")

    # Skip if file already exists (idempotency)
    if os.path.exists(parquet_filename):
        print(f"File {parquet_filename} already exists. Skipping.")
        return

    async with semaphore:  # Limit concurrent requests
        print(f"Requesting: {url}")
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"HTTP {response.status} for {url}. Data may not exist for this date.")
                    return
                
                # Read zip file bytes into memory
                zip_bytes = await response.read()
        except Exception as e:
            print(f"Network error on {date_str}: {e}")
            return

    # CPU-bound task: Offload to a separate thread to avoid blocking the async event loop
    await asyncio.to_thread(process_zip_to_parquet, zip_bytes, parquet_filename)

def process_zip_to_parquet(zip_bytes: bytes, parquet_filename: str):
    """
    Extracts the CSV from memory and converts it to a Parquet file.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, names=COLUMNS, header=None)

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df['price'] = df['price'].astype('float32')
        df['quantity'] = df['quantity'].astype('float32')
        
        df.to_parquet(parquet_filename, engine='pyarrow', compression='snappy')
        print(f"Saved: {parquet_filename} | Rows: {len(df)}")
    except Exception as e:
        print(f"Error processing data to Parquet: {e}")

async def main(symbol: str, start_date: str, end_date: str, output_dir: str = "data"):
    """
    Main orchestrator for batch downloading.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [(start + timedelta(days=x)).strftime("%Y-%m-%d") for x in range((end - start).days + 1)]
    
    # Restrict to 5 concurrent downloads to respect rate limits
    semaphore = asyncio.Semaphore(5) 
    
    async with aiohttp.ClientSession() as session:
        tasks = [download_and_process(session, symbol, date, output_dir, semaphore) for date in date_list]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Target: 3 months of BTCUSDT tick data
    SYMBOL = "BTCUSDT"
    START_DATE = "2025-10-01" 
    END_DATE = "2025-12-31" 
    
    # Run the async event loop
    asyncio.run(main(SYMBOL, START_DATE, END_DATE))