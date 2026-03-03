import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_walk_forward_model(data_path: str = "data/feature_matrix.parquet"):
    """
    Trains an XGBoost model to predict forward returns using walk-forward cross-validation.
    """
    print("Loading engineered features...")
    df = pd.read_parquet(data_path)
    
    # Define our features (X) and our label (y)
    features = ['obi', 'volume', 'buy_vol', 'sell_vol', 'volatility_20']
    target = 'target_fwd_return'
    
    X = df[features]
    y = df[target]
    
    # ---------------------------------------------------------
    # Walk-Forward Split (80% Train, 20% Out-of-Sample Test)
    # ---------------------------------------------------------
    split_idx = int(len(df) * 0.80)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} samples... Testing on {len(X_test)} out-of-sample periods.")

    # Initialize the XGBoost Regressor
    # Hyperparameters tuned for noisy financial time-series (low depth, conservative learning rate)
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training XGBoost Model...")
    model.fit(X_train, y_train)
    
    print("Evaluating Out-of-Sample Performance...")
    predictions = model.predict(X_test)
    
    # Performance Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Out-of-Sample MSE: {mse:.8f}")
    print(f"Out-of-Sample R-squared: {r2:.4f}")
    
    # Feature Importance
    importance = model.feature_importances_
    for name, imp in zip(features, importance):
        print(f"Feature: {name:15} | Importance: {imp:.4f}")
        
    # Save the predictions back to the test dataframe for the backtester
    results_df = pd.DataFrame({
        'close_price': df['close_price'].iloc[split_idx:],
        'actual_return': y_test,
        'predicted_return': predictions
    }, index=y_test.index)
    
    results_df.to_parquet("data/predictions.parquet")
    print("Predictions saved for backtesting.")

if __name__ == "__main__":
    train_walk_forward_model()