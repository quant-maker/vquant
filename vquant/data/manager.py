#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Manager - Manage cached market data
"""

import argparse
import logging
from vquant.data import (
    fetch_klines_multiple_batches,
    get_cached_klines,
    clear_cache,
    prefetch_all_data
)
from vquant.data.fetcher import _cache


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_cache():
    """List all cached data"""
    import sqlite3
    
    conn = sqlite3.connect(_cache.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT symbol, interval, 
               COUNT(*) as count,
               MIN(timestamp) as min_ts,
               MAX(timestamp) as max_ts
        FROM klines
        GROUP BY symbol, interval
        ORDER BY symbol, interval
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("No cached data found")
        return
    
    print(f"\n{'Symbol':<12} {'Interval':<10} {'Count':<8} {'Date Range'}")
    print("=" * 80)
    
    for row in results:
        symbol, interval, count, min_ts, max_ts = row
        
        from datetime import datetime
        min_date = datetime.fromtimestamp(min_ts / 1000).strftime('%Y-%m-%d')
        max_date = datetime.fromtimestamp(max_ts / 1000).strftime('%Y-%m-%d')
        
        print(f"{symbol:<12} {interval:<10} {count:<8} {min_date} to {max_date}")
    
    print()


def prefetch_data(symbol: str, interval: str, days: int):
    """Pre-fetch and cache data for future use"""
    print(f"\nPre-fetching {symbol} {interval} data for {days} days...")
    
    df = fetch_klines_multiple_batches(
        symbol=symbol,
        interval=interval,
        days=days,
        verbose=True,
        use_cache=True,
        force_refresh=False
    )
    
    if df is not None:
        print(f"✓ Successfully cached {len(df)} klines")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
    else:
        print("❌ Failed to fetch data")


def refresh_data(symbol: str, interval: str, days: int):
    """Refresh cached data (force re-fetch from API)"""
    print(f"\nRefreshing {symbol} {interval} data...")
    
    # Clear old cache
    clear_cache(symbol, interval)
    
    # Fetch new data
    df = fetch_klines_multiple_batches(
        symbol=symbol,
        interval=interval,
        days=days,
        verbose=True,
        use_cache=True,
        force_refresh=True
    )
    
    if df is not None:
        print(f"✓ Successfully refreshed {len(df)} klines")
    else:
        print("❌ Failed to refresh data")


def view_data(symbol: str, interval: str, days: int):
    """View cached data"""
    print(f"\nViewing {symbol} {interval} cached data (last {days} days)...")
    
    df = get_cached_klines(symbol, interval, days)
    
    if df is None:
        print(f"❌ No cached data found for {symbol} {interval}")
        return
    
    print(f"\n✓ Found {len(df)} klines")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    print(f"\nStatistics:")
    print(df[['open', 'high', 'low', 'close', 'volume']].describe())


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Manage cached market data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all cached data
  python -m vquant.data.manager list
  
  # Pre-fetch 1-minute data from 2023-01-01 for all symbols
  python -m vquant.data.manager prefetch --all --start-date 2023-01-01
  
  # Pre-fetch 1-minute data for specific symbol
  python -m vquant.data.manager prefetch --symbol BTCUSDC --start-date 2023-01-01
  
  # Adjust request delay to avoid rate limits (increase if getting errors)
  python -m vquant.data.manager prefetch --all --start-date 2023-01-01 --delay 1.0
  
  # View cached data
  python -m vquant.data.manager view --symbol BTCUSDC --interval 1m --days 30
  
  # Refresh data (force re-fetch)
  python -m vquant.data.manager refresh --symbol BTCUSDC --interval 1m --days 90
  
  # Clear cache for specific symbol
  python -m vquant.data.manager clear --symbol BTCUSDC --interval 1m
  
  # Clear all cache
  python -m vquant.data.manager clear --all

Note:
  - Only 1-minute data is prefetched (use resample_klines() to aggregate)
  - Request delay prevents rate limiting (default 0.5s, adjust if needed)
  - 1m data is large: ~43,200 bars/month, ~520,000 bars/year per symbol
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    subparsers.add_parser('list', help='List all cached data')
    
    # Prefetch command
    prefetch_parser = subparsers.add_parser('prefetch', help='Pre-fetch 1-minute data')
    prefetch_parser.add_argument('--symbol', help='Trading pair (optional, omit to fetch all)')
    prefetch_parser.add_argument('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD), default: 2023-01-01')
    prefetch_parser.add_argument('--delay', type=float, default=0.5, help='Request delay in seconds (default: 0.5)')
    prefetch_parser.add_argument('--all', action='store_true', help='Prefetch all symbols (1m data only)')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View cached data')
    view_parser.add_argument('--symbol', required=True, help='Trading pair')
    view_parser.add_argument('--interval', default='1h', help='K-line interval')
    view_parser.add_argument('--days', type=int, default=30, help='Days to view')
    
    # Refresh command
    refresh_parser = subparsers.add_parser('refresh', help='Refresh cached data')
    refresh_parser.add_argument('--symbol', required=True, help='Trading pair')
    refresh_parser.add_argument('--interval', default='1h', help='K-line interval')
    refresh_parser.add_argument('--days', type=int, default=90, help='Days to fetch')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cached data')
    clear_parser.add_argument('--symbol', help='Trading pair (optional)')
    clear_parser.add_argument('--interval', help='K-line interval (optional)')
    clear_parser.add_argument('--all', action='store_true', help='Clear all cached data')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_cache()
        
        elif args.command == 'prefetch':
            if args.all or not args.symbol:
                # Prefetch all symbols (1m data only)
                print(f"\n预拉取所有交易对的1分钟数据（从 {args.start_date} 至今）...")
                prefetch_all_data(start_date=args.start_date, request_delay=args.delay)
            else:
                # Prefetch specific symbol (1m data)
                print(f"\n预拉取 {args.symbol} 的1分钟数据（从 {args.start_date} 至今）...")
                prefetch_all_data(symbols=[args.symbol], start_date=args.start_date, request_delay=args.delay)
        
        elif args.command == 'view':
            view_data(args.symbol, args.interval, args.days)
        
        elif args.command == 'refresh':
            refresh_data(args.symbol, args.interval, args.days)
        
        elif args.command == 'clear':
            if args.all:
                print("\n⚠️  This will clear ALL cached data. Are you sure? (y/n): ", end='')
                response = input().strip().lower()
                if response == 'y':
                    clear_cache()
                    print("✓ All cached data cleared")
                else:
                    print("Cancelled")
            elif args.symbol:
                clear_cache(args.symbol, args.interval)
                if args.interval:
                    print(f"✓ Cleared cache for {args.symbol} {args.interval}")
                else:
                    print(f"✓ Cleared cache for {args.symbol}")
            else:
                print("❌ Please specify --symbol or --all")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
