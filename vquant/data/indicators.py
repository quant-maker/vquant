#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Technical Indicators Module - Unified technical indicator calculation and statistics preparation
Provides shared data preparation logic for all predictors
"""

import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def prepare_market_stats(df: pd.DataFrame, df_display: pd.DataFrame, 
                         ma_dict: Dict[int, pd.Series], args) -> Dict[str, Any]:
    """
    Calculate comprehensive market statistics and technical indicators
    
    Args:
        df: Full dataframe
        df_display: Display dataframe (limited rows)
        ma_dict: Moving average dictionary {period: series}
        args: Command line arguments containing configuration
        
    Returns:
        Statistics dictionary containing all market data and indicators
    """
    from vquant.model.vision import calculate_rsi, calculate_macd
    
    # Basic price statistics
    current_price = df_display.iloc[-1]["Close"]
    first_price = df_display.iloc[0]["Open"]
    price_change = current_price - first_price
    price_change_pct = (price_change / first_price) * 100
    high_price = df_display["High"].max()
    low_price = df_display["Low"].min()
    
    # Volume statistics
    total_volume = df_display["Volume"].sum()
    total_trades = df_display["Trades"].sum()
    total_taker_buy = df_display["TakerBuyBase"].sum()
    buy_ratio = (total_taker_buy / total_volume * 100) if total_volume > 0 else 0
    
    # Technical indicators - calculate on full data, then truncate for display
    rsi_full = calculate_rsi(df["Close"])
    macd_full, signal_full, histogram_full = calculate_macd(df["Close"])
    
    current_rsi = rsi_full.iloc[-1]
    current_macd = macd_full.iloc[-1]
    current_signal = signal_full.iloc[-1]
    
    # Truncate indicators for plotting
    rsi_display = rsi_full.iloc[-len(df_display):]
    macd_display = macd_full.iloc[-len(df_display):]
    signal_display = signal_full.iloc[-len(df_display):]
    histogram_display = histogram_full.iloc[-len(df_display):]
    
    # Market dynamics indicators
    volatility = df_display["Close"].pct_change().std() * 100
    
    # Momentum calculation
    recent_avg = df_display.iloc[-10:]["Close"].mean()
    earlier_avg = (
        df_display.iloc[-30:-10]["Close"].mean()
        if len(df_display) >= 30
        else df_display.iloc[:10]["Close"].mean()
    )
    momentum = (
        ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0
    )
    
    # Calculate impulse (momentum acceleration)
    impulse = _calculate_impulse(df_display, momentum)
    
    # Volume strength
    recent_volume = df_display.iloc[-10:]["Volume"].mean()
    avg_volume = df_display["Volume"].mean()
    volume_strength = (
        ((recent_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
    )
    
    # ATR (Average True Range)
    atr, atr_pct = _calculate_atr(df_display, current_price)
    
    # MA inflection point detection
    ma7_inflection = _detect_ma_inflection(ma_dict)
    
    # Build statistics dictionary
    stats = {
        # Price data
        "current_price": current_price,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "high": high_price,
        "low": low_price,
        
        # Volume data
        "total_volume": total_volume,
        "total_trades": total_trades,
        "buy_ratio": buy_ratio,
        
        # Technical indicators
        "rsi": current_rsi,
        "current_rsi": current_rsi,
        "macd": current_macd,
        "current_macd": current_macd,
        "macd_signal": current_signal,
        "current_signal": current_signal,
        
        # Market dynamics
        "volatility": volatility,
        "momentum": momentum,
        "impulse": impulse,
        "volume_strength": volume_strength,
        "atr": atr,
        "atr_pct": atr_pct,
        
        # Indicator series for plotting
        "rsi_series": rsi_display,
        "macd_series": macd_display,
        "signal_series": signal_display,
        "histogram_series": histogram_display,
        
        # Moving average data
        "ma_dict": {
            period: ma_series.iloc[-1] for period, ma_series in ma_dict.items()
        },
        "ma_periods": args.ma_periods,
        "ma7_inflection": ma7_inflection,
        
        # Recent K-line data for table display
        "recent_klines": df_display.iloc[-24:]
        .reset_index()[["timestamp", "Open", "High", "Low", "Close"]]
        .to_dict("records"),
        "recent_ma": {
            period: ma_series.iloc[-24:].tolist()
            for period, ma_series in ma_dict.items()
        },
    }
    
    # Add current MA values for predictor use
    if 7 in ma_dict:
        stats["current_ma7"] = ma_dict[7].iloc[-1]
    if 25 in ma_dict:
        stats["current_ma25"] = ma_dict[25].iloc[-1]
    if 99 in ma_dict:
        stats["current_ma99"] = ma_dict[99].iloc[-1]
    
    return stats


def _calculate_impulse(df_display: pd.DataFrame, momentum: float) -> float:
    """
    Calculate impulse (momentum acceleration) with volume confirmation
    
    Args:
        df_display: Display dataframe
        momentum: Current momentum value
        
    Returns:
        Impulse value (volume-weighted momentum change)
    """
    if len(df_display) >= 50:
        # Current momentum (already calculated)
        # Previous momentum (20 bars ago)
        prev_recent = df_display.iloc[-30:-20]["Close"].mean()
        prev_earlier = df_display.iloc[-50:-30]["Close"].mean()
        prev_momentum = ((prev_recent - prev_earlier) / prev_earlier * 100) if prev_earlier > 0 else 0
        
        # Impulse = change in momentum
        impulse = momentum - prev_momentum
        
        # Volume confirmation factor
        recent_volume_impulse = df_display.iloc[-1]["Volume"]
        avg_volume_impulse = df_display.iloc[-20:]["Volume"].mean()
        volume_factor = min(recent_volume_impulse / avg_volume_impulse if avg_volume_impulse > 0 else 1.0, 2.0)
        
        # Apply volume weighting
        impulse = impulse * volume_factor
    else:
        impulse = 0
    
    return impulse


def _calculate_atr(df_display: pd.DataFrame, current_price: float) -> tuple:
    """
    Calculate Average True Range (ATR) and its percentage
    
    Args:
        df_display: Display dataframe
        current_price: Current price
        
    Returns:
        Tuple of (atr, atr_pct)
    """
    high_low = df_display["High"] - df_display["Low"]
    high_close = abs(df_display["High"] - df_display["Close"].shift())
    low_close = abs(df_display["Low"] - df_display["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().iloc[-1]
    atr_pct = (atr / current_price * 100) if current_price > 0 else 0
    
    return atr, atr_pct


def _detect_ma_inflection(ma_dict: Dict[int, pd.Series]) -> str:
    """
    Detect MA7 inflection point within last 4 bars
    
    Args:
        ma_dict: Moving average dictionary
        
    Returns:
        "upward", "downward", or "continuing"
    """
    if 7 not in ma_dict or len(ma_dict[7]) < 5:
        return "continuing"
    
    ma7_series = ma_dict[7]
    
    # Check for inflection points in last 4 bars
    for i in range(1, 5):
        if len(ma7_series) >= i + 2:
            ma7_current = ma7_series.iloc[-i]
            ma7_prev1 = ma7_series.iloc[-i - 1]
            ma7_prev2 = ma7_series.iloc[-i - 2]
            
            slope_recent = ma7_current - ma7_prev1
            slope_before = ma7_prev1 - ma7_prev2
            
            if slope_before > 0 and slope_recent < 0:
                return "downward"
            elif slope_before < 0 and slope_recent > 0:
                return "upward"
    
    return "continuing"


def add_funding_rate_to_stats(stats: Dict[str, Any], funding_info: Dict, 
                               funding_times: List, funding_rates: List) -> None:
    """
    Add funding rate data to statistics dictionary (in-place)
    
    Args:
        stats: Statistics dictionary to update
        funding_info: Current funding rate info
        funding_times: Historical funding times
        funding_rates: Historical funding rates
    """
    if funding_info:
        stats["funding_rate"] = funding_info["rate"]
        stats["funding_next"] = funding_info["next_time"]
    
    if funding_times and funding_rates:
        stats["funding_history"] = (funding_times, funding_rates)
