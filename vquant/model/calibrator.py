#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Threshold Calibrator - Analyze historical data to calculate reasonable thresholds for technical indicators
Used for periodic updates of quantitative prediction model parameters
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from ..data import fetch_klines_multiple_batches


def calculate_ma(prices, period):
    """Calculate moving average"""
    return pd.Series(prices).rolling(window=period).mean()


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    deltas = pd.Series(prices).diff()
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    prices_series = pd.Series(prices)
    ema_fast = prices_series.ewm(span=fast).mean()
    ema_slow = prices_series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line


def calculate_impulse(closes, volumes):
    """
    Calculate Impulse - momentum acceleration with volume confirmation
    Impulse measures the change in momentum, indicating acceleration/deceleration
    
    Args:
        closes: List of closing prices
        volumes: List of volumes
    
    Returns:
        List of impulse values (percentage)
    """
    impulse_values = []
    
    # Need at least 50 bars for calculation
    for i in range(50, len(closes)):
        # Calculate current momentum (last 10 vs previous 20)
        current_recent = np.mean(closes[i-10:i])
        current_earlier = np.mean(closes[i-30:i-10])
        current_momentum = (current_recent - current_earlier) / current_earlier * 100 if current_earlier > 0 else 0
        
        # Calculate previous momentum (20 bars ago)
        prev_recent = np.mean(closes[i-30:i-20])
        prev_earlier = np.mean(closes[i-50:i-30])
        prev_momentum = (prev_recent - prev_earlier) / prev_earlier * 100 if prev_earlier > 0 else 0
        
        # Impulse = change in momentum
        impulse = current_momentum - prev_momentum
        
        # Volume confirmation factor (normalize volume change)
        avg_volume = np.mean(volumes[i-20:i])
        current_volume = volumes[i]
        volume_factor = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Apply volume weighting (cap at 2x)
        volume_factor = min(volume_factor, 2.0)
        weighted_impulse = impulse * volume_factor
        
        impulse_values.append(weighted_impulse)
    
    return impulse_values


def analyze_symbol(symbol, interval='1h', days=365):
    """Analyze all indicators for a single trading pair"""
    print(f"\n{'='*60}")
    print(f"Analyzing {symbol} ({interval}, {days} days data)")
    print(f"{'='*60}")
    
    # Fetch historical K-line data (in batches)
    df = fetch_klines_multiple_batches(symbol, interval=interval, days=days)
    if df is None or len(df) < 100:
        print(f"Insufficient data: {len(df) if df is not None else 0}")
        return None
    
    # Extract data and ensure numeric types
    closes = df['close'].astype(float).tolist()
    volumes = df['volume'].astype(float).tolist()
    
    # Calculate MA7
    ma7 = calculate_ma(closes, 7)
    
    # Calculate price deviation from MA7 percentage
    ma_deviations = []
    for i in range(len(closes)):
        if not pd.isna(ma7.iloc[i]) and ma7.iloc[i] > 0:
            deviation = (closes[i] - ma7.iloc[i]) / ma7.iloc[i] * 100
            ma_deviations.append(deviation)
    
    # Calculate momentum
    momentum_values = []
    for i in range(30, len(closes)):
        recent_avg = np.mean(closes[i-10:i])
        earlier_avg = np.mean(closes[i-30:i-10])
        if earlier_avg > 0:
            momentum = (recent_avg - earlier_avg) / earlier_avg * 100
            momentum_values.append(momentum)
    
    # Calculate impulse (momentum acceleration)
    impulse_values = calculate_impulse(closes, volumes)
    
    # Calculate RSI
    rsi_values = calculate_rsi(closes)
    rsi_values = rsi_values.dropna().tolist()
    
    # Calculate MACD difference
    macd, signal = calculate_macd(closes)
    macd_diffs = []
    for i in range(len(macd)):
        if not pd.isna(macd.iloc[i]) and not pd.isna(signal.iloc[i]):
            # 计算MACD差值相对于价格的百分比
            diff_pct = (macd.iloc[i] - signal.iloc[i]) / closes[i] * 100
            macd_diffs.append(diff_pct)
    
    # 计算成交量变化
    volume_changes = []
    window = 20
    for i in range(window, len(volumes)):
        avg_volume = np.mean(volumes[i-window:i])
        if avg_volume > 0:
            change = (volumes[i] - avg_volume) / avg_volume * 100
            volume_changes.append(change)
    
    results = {
        'symbol': symbol,
        'ma_deviation': ma_deviations,
        'momentum': momentum_values,
        'impulse': impulse_values,
        'rsi': rsi_values,
        'macd_diff': macd_diffs,
        'volume_change': volume_changes
    }
    
    return results


def print_statistics(data, name):
    """Print statistical information"""
    if not data:
        print(f"{name}: no data")
        return None
    
    data = np.array(data)
    stats = {
        'count': int(len(data)),
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'p5': float(np.percentile(data, 5)),
        'p10': float(np.percentile(data, 10)),
        'p25': float(np.percentile(data, 25)),
        'p50': float(np.percentile(data, 50)),
        'p75': float(np.percentile(data, 75)),
        'p90': float(np.percentile(data, 90)),
        'p95': float(np.percentile(data, 95))
    }
    
    print(f"\n{name} Statistics:")
    print(f"  Sample count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std dev: {stats['std']:.2f}")
    print(f"  Max: {stats['max']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  95th percentile: {stats['p95']:.2f}")
    print(f"  90th percentile: {stats['p90']:.2f}")
    print(f"  75th percentile: {stats['p75']:.2f}")
    print(f"  50th percentile: {stats['p50']:.2f}")
    print(f"  25th percentile: {stats['p25']:.2f}")
    print(f"  10th percentile: {stats['p10']:.2f}")
    print(f"  5th percentile: {stats['p5']:.2f}")
    
    return stats


def calculate_thresholds(result):
    """Calculate recommended thresholds based on statistics"""
    thresholds = {}
    
    # MA deviation thresholds
    ma_stats = print_statistics(result['ma_deviation'], "MA Deviation (%)")
    if ma_stats:
        thresholds['ma_deviation'] = {
            'extreme_bullish': ma_stats['p95'],  # +100分
            'strong_bullish': ma_stats['p90'],   # +80分
            'bullish': ma_stats['p75'],          # +50分
            'neutral_high': 0,                   # +20分
            'neutral_low': ma_stats['p25'],      # -20分
            'bearish': ma_stats['p10'],          # -50分
            'strong_bearish': ma_stats['p5'],    # -80分
        }
    
    # Momentum thresholds
    momentum_stats = print_statistics(result['momentum'], "Momentum (%)")
    if momentum_stats:
        thresholds['momentum'] = {
            'extreme_bullish': momentum_stats['p95'],
            'strong_bullish': momentum_stats['p90'],
            'bullish': momentum_stats['p75'],
            'neutral_high': 0,
            'neutral_low': momentum_stats['p25'],
            'bearish': momentum_stats['p10'],
            'strong_bearish': momentum_stats['p5'],
        }
    
    # Impulse thresholds (momentum acceleration)
    impulse_stats = print_statistics(result['impulse'], "Impulse (%)")
    if impulse_stats:
        thresholds['impulse'] = {
            'extreme_bullish': impulse_stats['p95'],
            'strong_bullish': impulse_stats['p90'],
            'bullish': impulse_stats['p75'],
            'neutral_high': 0,
            'neutral_low': impulse_stats['p25'],
            'bearish': impulse_stats['p10'],
            'strong_bearish': impulse_stats['p5'],
        }
    
    # RSI thresholds
    rsi_stats = print_statistics(result['rsi'], "RSI")
    if rsi_stats:
        thresholds['rsi'] = {
            'extreme_overbought': rsi_stats['p95'],  # -100分（超买看跌）
            'overbought': rsi_stats['p90'],          # -80分
            'slightly_overbought': rsi_stats['p75'], # -50分
            'neutral_high': rsi_stats['p50'],        # -20分
            'neutral_low': rsi_stats['p25'],         # +20分
            'slightly_oversold': rsi_stats['p10'],   # +50分
            'oversold': rsi_stats['p5'],             # +80分
        }
    
    # MACD difference thresholds
    macd_stats = print_statistics(result['macd_diff'], "MACD Diff (%)")
    if macd_stats:
        thresholds['macd_diff'] = {
            'extreme_bullish': macd_stats['p95'],
            'strong_bullish': macd_stats['p90'],
            'bullish': macd_stats['p75'],
            'neutral_high': 0,
            'neutral_low': macd_stats['p25'],
            'bearish': macd_stats['p10'],
            'strong_bearish': macd_stats['p5'],
        }
    
    # Volume change thresholds
    volume_stats = print_statistics(result['volume_change'], "Volume Change (%)")
    if volume_stats:
        thresholds['volume_change'] = {
            'extreme_volume': volume_stats['p95'],
            'high_volume': volume_stats['p90'],
            'above_average': volume_stats['p75'],
            'neutral': volume_stats['p50'],
            'below_average': volume_stats['p25'],
            'low_volume': volume_stats['p10'],
        }
    
    return thresholds


def save_thresholds(thresholds, symbol, output_dir='config'):
    """Save thresholds to JSON file"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    output_file = output_path / f'thresholds_{symbol.lower()}.json'
    
    data = {
        'symbol': symbol,
        'updated_at': datetime.now().isoformat(),
        'thresholds': thresholds
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nThresholds saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Calibrate technical indicator thresholds')
    parser.add_argument('--symbol', '-s', type=str, default='BTCUSDC',
                        help='Trading pair symbol (default: BTCUSDC)')
    parser.add_argument('--days', '-d', type=int, default=365,
                        help='Number of days to analyze (default: 365)')
    parser.add_argument('--interval', '-i', type=str, default='1h',
                        help='K-line period (default: 1h)')
    parser.add_argument('--output', '-o', type=str, default='config',
                        help='Output directory (default: config)')
    
    args = parser.parse_args()
    
    print(f"Starting calibration for {args.symbol} thresholds...")
    print(f"Parameters: interval={args.interval}, days={args.days}")
    
    # Analyze data
    result = analyze_symbol(args.symbol, interval=args.interval, days=args.days)
    
    if not result:
        print("Analysis failed, unable to fetch data")
        return 1
    
    # Calculate thresholds
    print(f"\n{'='*60}")
    print("Calculating Recommended Thresholds")
    print(f"{'='*60}")
    
    thresholds = calculate_thresholds(result)
    
    # Save to file
    output_file = save_thresholds(thresholds, args.symbol, args.output)
    
    print(f"\n✅ Done! View thresholds with:")
    print(f"   cat {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
