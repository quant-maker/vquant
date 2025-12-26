#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Universal training entry point for Kelly ML model
Supports training with 1m data aggregated to any interval
"""

import logging
import argparse
from pathlib import Path
from vquant.data import get_cached_klines, resample_klines, prepare_training_data
from vquant.analysis import KellyTrader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_kelly_model(
    symbol: str = "BTCUSDC",
    name: str = "test",
    interval: str = "1h",
    days: int = 180,
    lookforward_bars: int = 10,
    profit_threshold: float = 0.005,
    use_1m_data: bool = True
):
    """
    Train Kelly model using historical data
    
    Args:
        symbol: Trading pair
        name: Strategy name
        interval: Target K-line interval (1h, 4h, 1d)
        days: Days of historical data
        lookforward_bars: Bars to look ahead for labeling
        profit_threshold: Price change threshold (0.005 = 0.5%)
        use_1m_data: If True, load 1m data and resample; if False, load target interval directly
    """
    logger.info("="*60)
    logger.info(f"Training Kelly ML model for {symbol}")
    logger.info(f"Target interval: {interval} (use_1m_data={use_1m_data})")
    logger.info("="*60)
    
    # 1. Initialize Kelly trader
    logger.info(f"\n1. Initializing Kelly trader (name={name})...")
    trader = KellyTrader(symbol=symbol, name=name)
    logger.info(f"✓ Trader initialized")
    
    # 2. Load historical data
    if use_1m_data:
        logger.info(f"\n2. Loading {days} days of 1m data from cache...")
        df_1m = get_cached_klines(symbol, '1m', days=days)
        
        if df_1m is None or len(df_1m) < 100:
            logger.error(f"❌ No 1m data in cache. Please run:")
            logger.error(f"   python -m vquant.data.manager prefetch --symbol {symbol} --start-date <start-date>")
            return False
        
        logger.info(f"✓ Loaded {len(df_1m)} 1m klines from {df_1m.index.min()} to {df_1m.index.max()}")
        
        # Resample to target interval
        if interval != '1m':
            logger.info(f"\n3. Resampling 1m → {interval}...")
            df = resample_klines(df_1m, interval)
            logger.info(f"✓ Resampled to {len(df)} {interval} bars")
        else:
            df = df_1m
    else:
        logger.info(f"\n2. Loading {days} days of {interval} data from cache...")
        df = get_cached_klines(symbol, interval, days=days)
        
        if df is None or len(df) < 100:
            logger.error(f"❌ No {interval} data in cache")
            return False
        
        logger.info(f"✓ Loaded {len(df)} klines from {df.index.min()} to {df.index.max()}")
    
    if len(df) < 100:
        logger.error(f"❌ Insufficient data: {len(df)}")
        return False
    
    # 3. Prepare training data
    step_num = 3 if use_1m_data and interval != '1m' else 2
    logger.info(f"\n{step_num+1}. Preparing training data (lookforward={lookforward_bars}, threshold={profit_threshold:.2%})...")
    X, y = prepare_training_data(
        df=df,
        feature_extractor=trader._extract_features,
        lookforward_bars=lookforward_bars,
        profit_threshold=profit_threshold
    )
    
    if X is None or y is None:
        logger.error("❌ Failed to prepare training data")
        return False
    
    logger.info(f"✓ Prepared {len(y)} samples, positive rate: {y.mean():.2%}")
    
    # 4. Train model
    step_num = 4 if use_1m_data and interval != '1m' else 3
    logger.info(f"\n{step_num+1}. Training ML model...")
    
    # Initialize scaler and model if not exists
    if trader.scaler is None:
        from sklearn.preprocessing import StandardScaler
        trader.scaler = StandardScaler()
    
    if trader.model is None:
        from sklearn.ensemble import RandomForestClassifier
        trader.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
    
    # Scale features and train
    X_scaled = trader.scaler.fit_transform(X)
    trader.model.fit(X_scaled, y)
    
    # Evaluate
    train_score = trader.model.score(X_scaled, y)
    logger.info(f"✓ Model trained, accuracy: {train_score:.2%}")
    
    # Feature importance
    feature_importance = trader.model.feature_importances_
    feature_names = [
        'current_price', 'momentum_5', 'momentum_20', 'volatility',
        'volume_ratio', 'funding_rate', 'rsi', 'ma_deviation'
    ]
    logger.info("\nFeature importance:")
    for fname, importance in sorted(zip(feature_names, feature_importance), 
                                   key=lambda x: x[1], reverse=True):
        logger.info(f"  {fname:20s}: {importance:.4f}")
    
    # 5. Save model
    step_num = 5 if use_1m_data and interval != '1m' else 4
    logger.info(f"\n{step_num+1}. Saving model...")
    trader._save_model()
    logger.info(f"✓ Model saved to {trader.model_path}")
    
    # Update state
    trader.state['last_training_count'] = 0
    trader._save_state()
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info("="*60)
    logger.info(f"Model path: {trader.model_path}")
    logger.info(f"Scaler path: {trader.scaler_path}")
    logger.info(f"Training samples: {len(y)}")
    logger.info(f"Training accuracy: {train_score:.2%}")
    logger.info(f"Win rate (historical): {y.mean():.2%}")
    
    return True


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Train Kelly ML model using historical data',
        epilog="""
Examples:
  # Train with 1m data aggregated to 1h (recommended)
  python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 365
  
  # Train with 1m data aggregated to 4h
  python -m vquant.model.train --symbol ETHUSDC --interval 4h --days 365
  
  # Train directly from cached interval data (legacy)
  python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 180 --no-use-1m
  
  # Custom parameters
  python -m vquant.model.train --symbol SOLUSDC --interval 1d --days 730 --lookforward 5 --threshold 0.01
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDC',
        help='Trading pair symbol (default: BTCUSDC)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='test',
        help='Strategy name (default: test)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help='Target K-line interval (default: 1h)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Days of historical data (default: 365)'
    )
    parser.add_argument(
        '--lookforward',
        type=int,
        default=10,
        help='Bars to look forward for labeling (default: 10)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.005,
        help='Profit threshold for labeling (default: 0.005 = 0.5%%)'
    )
    parser.add_argument(
        '--no-use-1m',
        action='store_true',
        help='Load target interval directly instead of resampling from 1m (not recommended)'
    )
    
    args = parser.parse_args()
    
    success = train_kelly_model(
        symbol=args.symbol,
        name=args.name,
        interval=args.interval,
        days=args.days,
        lookforward_bars=args.lookforward,
        profit_threshold=args.threshold,
        use_1m_data=not args.no_use_1m
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
