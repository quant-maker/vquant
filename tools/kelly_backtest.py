#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kelly Strategy Backtesting Tool
Test trained Kelly model on historical data and generate PnL curves
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vquant.analysis.kelly import KellyTrader
from vquant.data.fetcher import fetch_klines_multiple_batches
from vquant.data.indicators import prepare_market_stats

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def prepare_test_data(symbol: str, interval: str, days: int, ma_periods: list) -> list:
    """
    Fetch and prepare test data
    
    Args:
        symbol: Trading symbol
        interval: K-line interval
        days: Number of days of data
        ma_periods: Moving average periods
        
    Returns:
        List of test samples with features and actual outcomes
    """
    logger.info(f"Fetching {days} days of data for {symbol} ({interval})...")
    
    # Fetch historical data
    df = fetch_klines_multiple_batches(
        symbol=symbol,
        interval=interval,
        days=days
    )
    
    if df is None or len(df) == 0:
        logger.error("Failed to fetch data")
        return []
    
    logger.info(f"Fetched {len(df)} bars")
    
    # Calculate moving averages
    ma_dict = {}
    for period in ma_periods:
        ma_dict[period] = df["Close"].rolling(window=period).mean()
    
    # Prepare test samples
    test_data = []
    lookback = max(ma_periods) + 50  # Need enough history for indicators
    
    for i in range(lookback, len(df) - 1):
        try:
            # Get data up to current point
            df_current = df.iloc[:i+1]
            df_display = df_current.iloc[-100:]  # Last 100 bars for display
            
            # Create args-like object for stats calculation
            class Args:
                def __init__(self):
                    self.ma_periods = ma_periods
                    self.limit = 100
            
            args = Args()
            
            # Calculate market stats
            ma_dict_current = {}
            for period in ma_periods:
                ma_dict_current[period] = df_current["Close"].rolling(window=period).mean()
            
            stats = prepare_market_stats(df_current, df_display, ma_dict_current, args)
            
            # Extract features (matching Kelly model feature extraction)
            features = extract_features(stats, df_current)
            
            # Calculate actual outcome (next bar)
            entry_price = df.iloc[i]["Close"]
            exit_price = df.iloc[i+1]["Close"]
            actual_pnl = ((exit_price - entry_price) / entry_price) * 100
            
            test_data.append({
                'features': features,
                'actual_pnl': actual_pnl,
                'timestamp': df.iloc[i]["timestamp"],
                'entry_price': entry_price,
                'exit_price': exit_price
            })
            
        except Exception as e:
            logger.warning(f"Failed to process bar {i}: {e}")
            continue
    
    logger.info(f"Prepared {len(test_data)} test samples")
    return test_data


def extract_features(stats: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Extract features matching Kelly model's feature extraction
    
    Args:
        stats: Market statistics dictionary
        df: Full dataframe
        
    Returns:
        Feature array
    """
    features = []
    
    # Price features
    current_price = stats.get('current_price', 0)
    features.append(current_price)
    
    # Momentum features
    if len(df) >= 20:
        momentum_5 = (df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) * 100
        momentum_20 = (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1) * 100
        features.extend([momentum_5, momentum_20])
    else:
        features.extend([0, 0])
    
    # Volatility
    volatility = stats.get('volatility', 0)
    features.append(volatility)
    
    # Volume
    if len(df) >= 20:
        recent_volume = df["Volume"].iloc[-5:].mean()
        avg_volume = df["Volume"].iloc[-20:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        features.append(volume_ratio)
    else:
        features.append(1.0)
    
    # Funding rate (if available)
    funding_rate = stats.get('funding_rate', 0)
    features.append(funding_rate)
    
    # RSI
    rsi = stats.get('current_rsi', 50)
    features.append(rsi)
    
    # MA deviation
    ma7 = stats.get('current_ma7', current_price)
    ma_deviation = ((current_price - ma7) / ma7 * 100) if ma7 > 0 else 0
    features.append(ma_deviation)
    
    # MACD
    macd = stats.get('current_macd', 0)
    features.append(macd)
    
    # ATR percentage
    atr_pct = stats.get('atr_pct', 0)
    features.append(atr_pct)
    
    return np.array(features)


def run_backtest(symbol: str, name: str, interval: str, days: int, 
                 ma_periods: list, save_plot: bool = True):
    """
    Run backtest and generate PnL curve
    
    Args:
        symbol: Trading symbol
        name: Strategy name
        interval: K-line interval
        days: Days of test data
        ma_periods: Moving average periods
        save_plot: Whether to save plot
    """
    logger.info("="*80)
    logger.info(f"Kelly Strategy Backtest")
    logger.info(f"Symbol: {symbol}, Name: {name}, Interval: {interval}")
    logger.info("="*80)
    
    # Initialize Kelly trader (loads trained model)
    try:
        trader = KellyTrader(
            symbol=symbol,
            name=name,
            config_dir="config"
        )
        logger.info("Kelly trader initialized")
    except Exception as e:
        logger.exception(f"Failed to initialize trader: {e}")
        return
    
    # Check if model is trained
    if trader.model is None:
        logger.error("="*80)
        logger.error("No trained model found!")
        logger.error("="*80)
        logger.error("")
        logger.error("To train the Kelly model, you need to:")
        logger.error("1. Run the strategy in live/paper trading mode to collect data")
        logger.error("   Example:")
        logger.error(f"   conda run -n gpt python main.py --symbol {symbol} --name {name} --predictor kelly --interval 1h")
        logger.error("")
        logger.error("2. The model will automatically train after collecting enough samples")
        logger.error(f"   - Minimum samples required: {trader.min_samples_for_training}")
        logger.error(f"   - Retraining interval: {trader.retrain_interval} trades")
        logger.error("")
        logger.error("3. Model files will be saved to:")
        logger.error(f"   - Model: data/kelly_model_{name}.pkl")
        logger.error(f"   - Scaler: data/kelly_scaler_{name}.pkl")
        logger.error("")
        logger.error("="*80)
        return
    
    # Prepare test data
    test_data = prepare_test_data(symbol, interval, days, ma_periods)
    
    if not test_data:
        logger.error("No test data available")
        return
    
    # Generate PnL curve
    if save_plot:
        plot_path = trader.plot_test_pnl_curve(test_data)
        if plot_path:
            logger.info(f"✓ PnL curve saved to: {plot_path}")
        else:
            logger.error("Failed to generate PnL curve")
    else:
        # Just calculate statistics
        actual_pnls = [s['actual_pnl'] for s in test_data]
        cumulative_pnl = np.sum(actual_pnls)
        win_rate = np.mean([1 if pnl > 0 else 0 for pnl in actual_pnls])
        
        logger.info(f"Test Results:")
        logger.info(f"  Total Trades: {len(test_data)}")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Cumulative PnL: {cumulative_pnl:.2f}%")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kelly Strategy Backtesting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDC",
        help="Trading symbol (default: BTCUSDC)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Strategy name (must match trained model)"
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="K-line interval (default: 1h)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of test data (default: 30)"
    )
    
    parser.add_argument(
        "--ma-periods",
        type=int,
        nargs="+",
        default=[7, 25, 99],
        help="Moving average periods (default: 7 25 99)"
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't save plot, just show statistics"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    try:
        run_backtest(
            symbol=args.symbol,
            name=args.name,
            interval=args.interval,
            days=args.days,
            ma_periods=args.ma_periods,
            save_plot=not args.no_plot
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
