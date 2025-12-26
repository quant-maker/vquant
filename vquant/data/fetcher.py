#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Data Fetcher - Fetch and cache historical market data
Stores data in SQLite database for reuse across modules
"""

import time
import logging
import sqlite3
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


class DataCache:
    """SQLite database cache for market data"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create klines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                close_time INTEGER,
                quote_volume REAL,
                trades INTEGER,
                taker_buy_base REAL,
                taker_buy_quote REAL,
                created_at INTEGER NOT NULL,
                UNIQUE(symbol, interval, timestamp)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_interval_timestamp 
            ON klines(symbol, interval, timestamp)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_cached_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data from database
        
        Args:
            symbol: Trading pair
            interval: K-line interval
            start_time: Start timestamp (milliseconds), None for all
            end_time: End timestamp (milliseconds), None for all
            
        Returns:
            DataFrame or None if no data found
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, open, high, low, close, volume,
                   close_time, quote_volume, trades, 
                   taker_buy_base, taker_buy_quote
            FROM klines
            WHERE symbol = ? AND interval = ?
        """
        params = [symbol, interval]
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp ASC"
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if len(df) == 0:
                return None
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Loaded {len(df)} cached klines for {symbol} {interval}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            conn.close()
            return None
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """
        Save data to database
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair
            interval: K-line interval
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        created_at = int(time.time() * 1000)
        saved_count = 0
        skipped_count = 0
        
        for idx, row in df.iterrows():
            timestamp = int(idx.timestamp() * 1000)
            
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO klines (
                        symbol, interval, timestamp, open, high, low, close, volume,
                        close_time, quote_volume, trades, 
                        taker_buy_base, taker_buy_quote, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, interval, timestamp,
                    float(row['open']), float(row['high']), 
                    float(row['low']), float(row['close']), float(row['volume']),
                    int(row.get('close_time', timestamp)),
                    float(row.get('quote_volume', 0)),
                    int(row.get('trades', 0)),
                    float(row.get('taker_buy_base', 0)),
                    float(row.get('taker_buy_quote', 0)),
                    created_at
                ))
                
                if cursor.rowcount > 0:
                    saved_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to insert row at {idx}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(
            f"Saved {saved_count} new klines, skipped {skipped_count} duplicates "
            f"for {symbol} {interval}"
        )
    
    def get_data_range(self, symbol: str, interval: str) -> Optional[Tuple[datetime, datetime]]:
        """Get the date range of cached data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM klines
            WHERE symbol = ? AND interval = ?
        """, (symbol, interval))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] and result[1]:
            min_ts = pd.to_datetime(result[0], unit='ms')
            max_ts = pd.to_datetime(result[1], unit='ms')
            return min_ts, max_ts
        
        return None
    
    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None):
        """Clear cached data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if symbol and interval:
            cursor.execute("DELETE FROM klines WHERE symbol = ? AND interval = ?", 
                         (symbol, interval))
            logger.info(f"Cleared cache for {symbol} {interval}")
        elif symbol:
            cursor.execute("DELETE FROM klines WHERE symbol = ?", (symbol,))
            logger.info(f"Cleared cache for {symbol}")
        else:
            cursor.execute("DELETE FROM klines")
            logger.info("Cleared all cached data")
        
        conn.commit()
        conn.close()


# Global cache instance
_cache = DataCache()


def fetch_klines_multiple_batches(
    symbol: str,
    interval: str = '1h',
    days: int = 365,
    batch_size: int = 1000,
    verbose: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    request_delay: float = 0.2
) -> Optional[pd.DataFrame]:
    """
    Fetch multiple days of K-line data in batches with database caching
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDC')
        interval: K-line period ('1m', '5m', '15m', '1h', '4h', '1d')
        days: Number of days needed
        batch_size: Number of klines per batch (max 1000 for Binance)
        verbose: Print progress messages
        use_cache: Use cached data if available
        force_refresh: Force fetch from API even if cache exists
        request_delay: Delay between API requests in seconds (default 0.2, increase for rate limit protection)
    
    Returns:
        Combined DataFrame with OHLCV data, or None if fetch fails
    """
    # Check cache first
    if use_cache and not force_refresh:
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        cached_df = _cache.get_cached_data(symbol, interval, start_time, end_time)
        
        if cached_df is not None and len(cached_df) > 0:
            # Check if we have enough data
            interval_hours = _get_interval_hours(interval)
            expected_klines = int(days * 24 / interval_hours)
            
            if len(cached_df) >= expected_klines * 0.8:  # 80% coverage
                if verbose:
                    logger.info(
                        f"Using cached data: {len(cached_df)} klines "
                        f"({cached_df.index.min()} to {cached_df.index.max()})"
                    )
                return cached_df
            else:
                if verbose:
                    logger.info(
                        f"Cached data insufficient ({len(cached_df)}/{expected_klines}), "
                        f"fetching from API..."
                    )
    
    # Fetch from API
    all_dfs = []
    interval_hours = _get_interval_hours(interval)
    hours_needed = days * 24
    klines_needed = int(hours_needed / interval_hours)
    batches = (klines_needed + batch_size - 1) // batch_size
    
    if verbose:
        logger.info(
            f"Fetching {symbol} {interval} data: "
            f"~{klines_needed} klines in {batches} batches"
        )
    
    # Start from current time and fetch backwards
    end_time = int(time.time() * 1000)
    
    for i in range(batches):
        if verbose:
            logger.info(f"Fetching batch {i+1}/{batches}...")
        
        try:
            url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': batch_size,
                'endTime': end_time
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                if verbose:
                    logger.warning(f"Batch {i+1}: No data returned")
                break
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
            
            all_dfs.append(df)
            
            if verbose:
                logger.info(f"Batch {i+1}: Success ({len(df)} klines)")
            
            # Update end time to earliest time of this batch
            end_time = int(df.index.min().timestamp() * 1000) - 1
            
            # Delay to avoid hitting rate limits (especially important for 1m data)
            if i < batches - 1:
                time.sleep(request_delay)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch {i+1}: Request failed: {e}")
            break
        except Exception as e:
            logger.error(f"Batch {i+1}: Unexpected error: {e}")
            break
    
    if not all_dfs:
        logger.error("No data fetched")
        return None
    
    # Merge all data and remove duplicates
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()
    
    if verbose:
        logger.info(
            f"Combined total: {len(combined_df)} klines "
            f"(≈ {len(combined_df) * interval_hours / 24:.1f} days)"
        )
    
    # Save to cache
    if use_cache:
        _cache.save_data(combined_df, symbol, interval)
    
    return combined_df


def _get_interval_hours(interval: str) -> float:
    """Convert interval string to hours"""
    interval_hours = {
        '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60, '30m': 30/60,
        '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12,
        '1d': 24, '3d': 72, '1w': 168
    }
    return interval_hours.get(interval, 1)


def get_cached_klines(
    symbol: str,
    interval: str,
    days: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Get cached klines from database
    
    Args:
        symbol: Trading pair
        interval: K-line interval
        days: Number of days (None for all cached data)
        
    Returns:
        DataFrame or None
    """
    end_time = int(time.time() * 1000)
    start_time = None
    
    if days is not None:
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    return _cache.get_cached_data(symbol, interval, start_time, end_time)


def clear_cache(symbol: Optional[str] = None, interval: Optional[str] = None):
    """
    Clear cached data
    
    Args:
        symbol: Trading pair (None for all)
        interval: K-line interval (None for all)
    """
    _cache.clear_cache(symbol, interval)


def resample_klines(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample 1-minute klines to higher timeframes
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        target_interval: Target interval ('1h', '4h', '1d')
    
    Returns:
        Resampled DataFrame
    """
    # Convert interval to pandas offset
    interval_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '12h': '12h',
        '1d': '1D',
        '1w': '1W'
    }
    
    if target_interval not in interval_map:
        raise ValueError(f"Unsupported interval: {target_interval}")
    
    freq = interval_map[target_interval]
    
    # Resample OHLCV data
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum',
        'taker_buy_base': 'sum',
        'taker_buy_quote': 'sum'
    })
    
    # Drop rows with NaN (incomplete periods)
    resampled = resampled.dropna()
    
    logger.info(f"Resampled from 1m to {target_interval}: {len(df)} -> {len(resampled)} bars")
    
    return resampled


def prepare_training_data(
    df: pd.DataFrame,
    feature_extractor,
    lookforward_bars: int = 10,
    profit_threshold: float = 0.005
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare training data from historical klines
    
    Args:
        df: DataFrame with OHLC data
        feature_extractor: Function to extract features from market data
            Should accept dict with 'current_price', 'klines', 'funding_rate'
            Should return numpy array of features
        lookforward_bars: Number of bars to look ahead for labeling
        profit_threshold: Price change threshold to consider profitable (0.005 = 0.5%)
    
    Returns:
        (X, y) tuple where:
            X: numpy array of features (n_samples, n_features)
            y: numpy array of labels (n_samples,) - 1 for profitable, 0 for not
    """
    X = []
    y = []
    
    closes = df['close'].values
    
    # Iterate through data, leaving enough bars for lookforward
    for i in range(len(df) - lookforward_bars):
        # Prepare market data for feature extraction
        # Get klines up to current point
        klines_slice = df.iloc[:i+1].tail(100)  # Last 100 bars for features
        
        if len(klines_slice) < 30:  # Need minimum history
            continue
        
        klines_list = []
        for idx, row in klines_slice.iterrows():
            klines_list.append([
                int(idx.timestamp() * 1000),  # timestamp
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ])
        
        market_data = {
            'current_price': float(closes[i]),
            'klines': klines_list,
            'funding_rate': 0.0001  # Default value, can be enhanced
        }
        
        # Extract features
        try:
            features = feature_extractor(market_data)
            if features is None:
                continue
            features_flat = features.flatten()
        except Exception as e:
            logger.warning(f"Feature extraction failed at index {i}: {e}")
            continue
        
        # Label: check if price goes up in next N bars
        future_price = closes[i + lookforward_bars]
        current_price = closes[i]
        price_change = (future_price - current_price) / current_price
        
        label = 1 if price_change >= profit_threshold else 0
        
        X.append(features_flat)
        y.append(label)
    
    if len(X) == 0:
        logger.error("No training samples generated")
        return None, None
    
    X = np.vstack(X)
    y = np.array(y)
    
    logger.info(
        f"Prepared {len(y)} training samples, "
        f"positive rate: {np.mean(y):.2%}"
    )
    
    return X, y


def prefetch_all_data(
    symbols: list = None,
    start_date: str = '2023-01-01',
    request_delay: float = 0.5
):
    """
    Prefetch 1-minute historical data for multiple symbols
    Data can be aggregated to 1h, 4h, or 1d during training
    
    Args:
        symbols: List of trading pairs, default: ['BTCUSDC', 'ETHUSDC', 'BNBUSDC', 'SOLUSDC', 'DOGEUSDC']
        start_date: Start date in format 'YYYY-MM-DD', default: '2023-01-01'
        request_delay: Delay between API requests in seconds, default: 0.5 (to avoid rate limits)
    
    Returns:
        dict: Statistics of prefetched data
    """
    # Default configurations
    if symbols is None:
        symbols = ['BTCUSDC', 'ETHUSDC', 'BNBUSDC', 'SOLUSDC', 'DOGEUSDC']
    
    # Calculate days from start_date to now
    from datetime import datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    now_dt = datetime.now()
    days = (now_dt - start_dt).days
    
    total_tasks = len(symbols)
    completed = 0
    failed = []
    stats = {
        'total': total_tasks,
        'success': 0,
        'failed': 0,
        'tasks': [],
        'start_date': start_date,
        'days': days,
        'interval': '1m'
    }
    
    logger.info("="*60)
    logger.info(f"开始预拉取1分钟数据：{len(symbols)}个交易对")
    logger.info(f"日期范围：{start_date} 至今 ({days}天)")
    logger.info(f"请求延迟：{request_delay}秒/批次 (防止触发限流)")
    logger.info("="*60)
    logger.warning("注意：1分钟数据量很大，请耐心等待...")
    
    for symbol in symbols:
        completed += 1
        interval = '1m'
        task_name = f"{symbol} {interval}"
        task_info = {
            'symbol': symbol,
            'interval': interval,
            'days': days,
            'start_date': start_date,
            'status': 'pending'
        }
        
        logger.info(f"\n[{completed}/{total_tasks}] 拉取 {task_name} 从{start_date}至今的数据...")
        expected_klines = days * 24 * 60  # 1分钟数据点数
        expected_batches = (expected_klines + 999) // 1000
        logger.info(f"预计：约{expected_klines:,}条数据，{expected_batches}个批次")
        
        try:
            start_time = time.time()
            
            df = fetch_klines_multiple_batches(
                symbol=symbol,
                interval=interval,
                days=days,
                verbose=True,
                use_cache=True,
                force_refresh=False,
                request_delay=request_delay  # 添加请求延迟
            )
            
            elapsed = time.time() - start_time
            
            if df is not None and len(df) > 0:
                task_info.update({
                    'status': 'success',
                    'count': len(df),
                    'elapsed': elapsed,
                    'start_date': str(df.index.min().date()),
                    'end_date': str(df.index.max().date())
                })
                
                logger.info(
                    f"✓ {task_name} 完成: {len(df):,}条数据, "
                    f"耗时{elapsed:.2f}秒 (约{elapsed/60:.1f}分钟), "
                    f"日期范围: {df.index.min().date()} 至 {df.index.max().date()}"
                )
                stats['success'] += 1
            else:
                task_info['status'] = 'no_data'
                logger.warning(f"⚠️  {task_name} 未获取到数据")
                failed.append(task_name)
                stats['failed'] += 1
            
            # 每个交易对之间额外延迟，避免请求过于频繁
            if completed < total_tasks:
                extra_delay = 2.0  # 交易对之间额外延迟2秒
                logger.info(f"等待{extra_delay}秒后继续下一个交易对...")
                time.sleep(extra_delay)
                
        except Exception as e:
            task_info['status'] = 'error'
            task_info['error'] = str(e)
            logger.error(f"❌ {task_name} 失败: {e}")
            failed.append(task_name)
            stats['failed'] += 1
        
        stats['tasks'].append(task_info)
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("数据拉取完成")
    logger.info("="*60)
    logger.info(f"总任务数: {total_tasks}")
    logger.info(f"成功: {stats['success']}")
    logger.info(f"失败: {stats['failed']}")
    
    if failed:
        logger.warning("\n失败的任务:")
        for task in failed:
            logger.warning(f"  - {task}")
    
    # 显示数据库统计
    logger.info("\n" + "="*60)
    logger.info("数据库统计")
    logger.info("="*60)
    
    try:
        conn = sqlite3.connect(_cache.db_path)
        cursor = conn.cursor()
        
        # 总体统计
        cursor.execute("SELECT COUNT(*) FROM klines")
        total_count = cursor.fetchone()[0]
        
        logger.info(f"总K线数: {total_count:,}")
        
        # 按交易对统计
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
        
        logger.info(f"\n{'交易对':<12} {'周期':<6} {'数量':<8} {'日期范围'}")
        logger.info("-" * 60)
        
        for row in results:
            symbol, interval, count, min_ts, max_ts = row
            min_date = datetime.fromtimestamp(min_ts / 1000).strftime('%Y-%m-%d')
            max_date = datetime.fromtimestamp(max_ts / 1000).strftime('%Y-%m-%d')
            logger.info(f"{symbol:<12} {interval:<6} {count:<8,} {min_date} 至 {max_date}")
        
        # 数据库大小
        import os
        db_size = os.path.getsize(_cache.db_path) / 1024 / 1024
        logger.info(f"\n数据库大小: {db_size:.2f} MB")
        
        stats['db_stats'] = {
            'total_count': total_count,
            'db_size_mb': db_size,
            'details': results
        }
        
        conn.close()
        
    except Exception as e:
        logger.error(f"无法获取数据库统计: {e}")
    
    return stats


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'prefetch':
        # Prefetch mode
        logger.info("Running in prefetch mode...")
        try:
            stats = prefetch_all_data()
            logger.info("\n✅ 全部完成！现在可以进行模型训练和验证了。")
        except KeyboardInterrupt:
            logger.warning("\n⚠️  用户中断")
        except Exception as e:
            logger.error(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Test mode
        print("Testing data fetcher with cache...")
        
        # First fetch (from API)
        print("\n1. First fetch (should use API):")
        df1 = fetch_klines_multiple_batches('BTCUSDC', interval='1h', days=7, verbose=True)
        
        if df1 is not None:
            print(f"   Fetched {len(df1)} klines")
        
        # Second fetch (from cache)
        print("\n2. Second fetch (should use cache):")
        df2 = fetch_klines_multiple_batches('BTCUSDC', interval='1h', days=7, verbose=True)
        
        if df2 is not None:
            print(f"   Fetched {len(df2)} klines")
        
        # Get cache info
        print("\n3. Cache info:")
        data_range = _cache.get_data_range('BTCUSDC', '1h')
        if data_range:
            print(f"   Cached data range: {data_range[0]} to {data_range[1]}")
        
        # Direct cache access
        print("\n4. Direct cache access:")
        df3 = get_cached_klines('BTCUSDC', '1h', days=3)
        if df3 is not None:
            print(f"   Retrieved {len(df3)} klines from cache")
        
        print("\n\nTo prefetch all data, run: python -m vquant.data.fetcher prefetch")
