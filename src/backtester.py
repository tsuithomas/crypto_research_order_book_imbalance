import pandas as pd
import numpy as np

def run_backtest(predictions_path: str = "data/predictions.parquet", threshold: float = 0.0015, fee: float = 0.0001):
    """
    Simulates a risk-managed statistical arbitrage strategy.
    threshold: 15 bps minimum predicted move to trigger a trade.
    fee: 1 bps maker/limit order fee assumption.
    """
    print("Loading out-of-sample predictions...")
    df = pd.read_parquet(predictions_path)
    
    # 1. Generate Strict Trading Signals
    df['signal'] = 0
    # Only trade if the predicted return is significantly higher than transaction costs
    df.loc[df['predicted_return'] > threshold, 'signal'] = 1
    df.loc[df['predicted_return'] < -threshold, 'signal'] = -1
    
    # Optional Risk Management: Prevent rapid flipping (whipsawing)
    # We hold the signal for at least 2 periods unless a strong opposite signal occurs
    df['signal'] = df['signal'].replace(0, np.nan).ffill(limit=1).fillna(0)    
    # 2. Calculate Strategy Returns
    df['strategy_return'] = df['signal'].shift(1) * df['actual_return']
    
    # 3. Apply Transaction Costs
    df['position_change'] = df['signal'].diff().abs()
    df['net_return'] = df['strategy_return'] - (df['position_change'] * fee)
    
    # 4. Calculate Equity Curve
    df['cumulative_market'] = np.exp(df['actual_return'].cumsum())
    df['cumulative_strategy'] = np.exp(df['net_return'].cumsum())
    
    # 5. Performance Metrics
    annualization_factor = np.sqrt(525600)
    
    mean_return = df['net_return'].mean()
    std_return = df['net_return'].std()
    
    if std_return > 0:
        sharpe_ratio = (mean_return / std_return) * annualization_factor
    else:
        sharpe_ratio = 0.0
        
    total_trades = df['position_change'].sum() / 2 
    
    print("\n--- Iteration 2: Risk-Managed Backtest ---")
    print(f"Total Trades Executed: {total_trades:,.0f}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Market Return (Buy & Hold): {(df['cumulative_market'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Strategy Net Return: {(df['cumulative_strategy'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Win Rate (Gross): {(df[df['strategy_return'] > 0].shape[0] / df[df['strategy_return'] != 0].shape[0] * 100):.2f}%")

if __name__ == "__main__":
    run_backtest()