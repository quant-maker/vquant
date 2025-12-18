#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import sys
import logging
import argparse
import pandas as pd

from dotenv import load_dotenv
from vquant.vision.chart import (
    fetch_binance_klines, 
    fetch_funding_rate, 
    fetch_funding_rate_history,
    plot_candlestick, 
    calculate_rsi, 
    calculate_macd)
from vquant.analysis.advisor import PositionAdvisor


# 配置日志
logger = logging.getLogger(__name__)
EPILOG = """
Examples:
  # Analysis only (default)
  %(prog)s --symbol BTCUSDC --interval 1h --service qwen
  
  # Analysis and execute trade
  %(prog)s --symbol BTCUSDC --interval 1h --service copilot --model gpt-4o --trade
  
  # Enable debug logging
  %(prog)s --symbol BTCUSDC --interval 4h --verbose --log-file logs/trading.log
"""


def setup_logging(args):
    # Configure log level
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.ERROR
    """Configure logging system"""
    log_format = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if args.log_file == "auto":
        args.log_file = f"logs/{args.symbol}_{args.interval}.log"
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or '.', exist_ok=True)
        handlers = [logging.FileHandler(args.log_file, encoding='utf-8')]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=level, format=log_format, datefmt=date_format, handlers=handlers)


def run(args):
    # main logical
    model_display = f"{args.service.upper()}"
    if args.service == 'copilot' and args.model:
        model_display = f"GitHub Copilot ({args.model})"
    logger.info(f"AI Service: {model_display}")
    if args.trade:
        logger.warning("Trading execution mode enabled")
    # Create charts directory
    os.makedirs('charts', exist_ok=True)
    # 1. Fetch K-line data
    logger.info("Step 1/4: Fetching K-line data...")
    extra_data = max(args.ma_periods) - 1
    df = fetch_binance_klines(
        symbol=args.symbol, interval=args.interval,
        limit=args.limit, extra_data=extra_data)
    if df is None:
        logger.error("Failed to fetch data")
        return None
    logger.info(f"Successfully fetched {len(df)} data points")
    # 2. Fetch funding rate
    logger.info("Step 2/4: Fetching funding rate...")
    funding_info = fetch_funding_rate(symbol=args.symbol)
    funding_times, funding_rates = fetch_funding_rate_history(symbol=args.symbol, limit=30)
    if funding_info:
        logger.info(f"Current funding rate: {funding_info['rate']:.4f}%")
    # 3. Calculate indicators and generate chart
    logger.info("Step 3/4: Calculating technical indicators and generating chart...")
    # 计算均线
    ma_dict = {}
    for period in args.ma_periods:
        ma_dict[period] = df['Close'].rolling(window=period).mean()
    # 只显示最后limit条数据
    df_display = df.iloc[-args.limit:].copy()
    ma_dict_display = {}
    for period, ma_series in ma_dict.items():
        ma_dict_display[period] = ma_series.iloc[-args.limit:]
    # 计算统计数据
    current_price = df_display.iloc[-1]['Close']
    first_price = df_display.iloc[0]['Open']
    price_change = current_price - first_price
    price_change_pct = (price_change / first_price) * 100
    high_price = df_display['High'].max()
    low_price = df_display['Low'].min()
    total_volume = df_display['Volume'].sum()
    total_trades = df_display['Trades'].sum()
    total_taker_buy = df_display['TakerBuyBase'].sum()
    buy_ratio = (total_taker_buy / total_volume * 100) if total_volume > 0 else 0
    # 技术指标 - 在完整数据上计算，然后截断用于显示
    rsi_full = calculate_rsi(df['Close'])
    macd_full, signal_full, histogram_full = calculate_macd(df['Close'])
    current_rsi = rsi_full.iloc[-1]
    current_macd = macd_full.iloc[-1]
    current_signal = signal_full.iloc[-1]
    # 截断指标数据用于绘图
    rsi_display = rsi_full.iloc[-args.limit:]
    macd_display = macd_full.iloc[-args.limit:]
    signal_display = signal_full.iloc[-args.limit:]
    histogram_display = histogram_full.iloc[-args.limit:]
    # 市场动态指标
    volatility = df_display['Close'].pct_change().std() * 100
    recent_avg = df_display.iloc[-10:]['Close'].mean()
    earlier_avg = df_display.iloc[-30:-10]['Close'].mean() if len(df_display) >= 30 else df_display.iloc[:10]['Close'].mean()
    momentum = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0
    recent_volume = df_display.iloc[-10:]['Volume'].mean()
    avg_volume = df_display['Volume'].mean()
    volume_strength = ((recent_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
    # ATR
    high_low = df_display['High'] - df_display['Low']
    high_close = abs(df_display['High'] - df_display['Close'].shift())
    low_close = abs(df_display['Low'] - df_display['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().iloc[-1]
    atr_pct = (atr / current_price * 100) if current_price > 0 else 0
    stats = {
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'high': high_price,
        'low': low_price,
        'total_volume': total_volume,
        'total_trades': total_trades,
        'buy_ratio': buy_ratio,
        'rsi': current_rsi,
        'macd': current_macd,
        'macd_signal': current_signal,
        'volatility': volatility,
        'momentum': momentum,
        'volume_strength': volume_strength,
        'atr': atr,
        'atr_pct': atr_pct,
        # 添加完整的指标序列用于绘图
        'rsi_series': rsi_display,
        'macd_series': macd_display,
        'signal_series': signal_display,
        'histogram_series': histogram_display
    }
    if funding_info:
        stats['funding_rate'] = funding_info['rate']
        stats['funding_next'] = funding_info['next_time']
    if funding_times and funding_rates:
        stats['funding_history'] = (funding_times, funding_rates)
    # 生成图表 - quiet模式下使用内存，否则保存文件
    if args.quiet:
        logger.info("Quiet mode: generating chart in memory (no file I/O)")
        image_bytes = plot_candlestick(
            df_display, symbol=args.symbol, save_path=None,
            ma_dict=ma_dict_display, stats=stats, return_bytes=True)
        save_path = None
    else:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'charts/{args.symbol}_{args.interval}_{timestamp}.png'
        plot_candlestick(
            df_display, symbol=args.symbol, save_path=save_path,
            ma_dict=ma_dict_display, stats=stats, return_bytes=False)
        logger.info(f"Chart saved: {save_path}")
        image_bytes = None
    # 4. AI analysis
    logger.info(f"Step 4/4: Performing AI analysis with {model_display}...")
    try:
        advisor = PositionAdvisor(service=args.service, model=args.model)
        result = advisor.analyze(
            image_path=save_path, 
            image_bytes=image_bytes, 
            symbol=args.symbol, 
            interval=args.interval, 
            current_price=current_price)
        # Execute trade if enabled
        trader = None
        if args.trade and result:
            logger.info("Executing trade...")
            try:
                from vquant.executor.trader import Trader
                trader = Trader(name=args.name, account=args.account)
                trader.trade(result, args)
                logger.info("Trade execution completed")
            except Exception as trade_error:
                logger.exception(f"Trade execution failed: {trade_error}", exc_info=True)
        return True
    except Exception as e:
        logger.error(f"AI analysis failed: {e}", exc_info=True)
        logger.info("Tip: Please ensure you have set the correct API key environment variables")
        logger.info("  - GitHub Copilot: GITHUB_TOKEN")
        logger.info("  - OpenAI: OPENAI_API_KEY")
        logger.info("  - Qwen: DASHSCOPE_API_KEY")
        logger.info("  - DeepSeek: DEEPSEEK_API_KEY")
        return {'chart_path': save_path, 'stats': stats, 'analysis': None}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Quantitative Trading System - K-line Analysis and Auto Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)
    # Basic parameters
    parser.add_argument(
        '--symbol', type=str, default='BTCUSDC',
        help='Trading pair symbol (default: BTCUSDC)')
    parser.add_argument(
        '--interval', type=str, default='1h',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
        help='K-line period (default: 1h)')
    # AI service configuration
    parser.add_argument(
        '--service', '-s', type=str, default='qwen',
        choices=['copilot', 'openai', 'qwen', 'deepseek'],
        help='AI service provider (default: copilot)')
    parser.add_argument(
        '--model', '-m', type=str,
        help='AI model name (copilot options: gpt-4o, claude-3.5-sonnet, o1-preview, etc.)')
    # Trading parameters
    parser.add_argument(
        '--trade', '-t', action='store_true',
        help='Enable trading execution mode (default: analysis only)')
    parser.add_argument(
        '--volume', type=float, default=0, help='max trading volumes(default: 0.0)')
    parser.add_argument(
        '--threshold', type=float, default=0.5, # (0,1)
        help='using with volume, > threshold then xxx < -theshold then xxx (default: 0.5)')
    parser.add_argument(
        '--account', '-a', type=str, default='li',
        help='Trading account name (default: li)')
    parser.add_argument(
        '--name', type=str, required=True,
        help='Strategy name (must be unique, prevents duplicate instances) (default: default)')
    # Technical parameters
    parser.add_argument(
        '--limit', '-l', type=int, default=72,
        help='Number of K-line data points (default: 72)')
    parser.add_argument(
        '--ma-periods', type=int, nargs='+', default=[7, 25, 99],
        help='Moving average periods (default: 7 25 99)')
    # Logging configuration
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Enable verbose logging output')
    parser.add_argument(
        '--log-file', type=str, help='Log file path')
    parser.add_argument(
        '--quiet', '-q', action='store_true', help='Quiet mode, only output errors')
    return parser.parse_args()


def main():
    """Command line entry point"""
    # Load environment variables
    load_dotenv()
    # Parse command line arguments
    args = parse_arguments()
    # Initialize logging system
    setup_logging(args)
    logger.info("="*60)
    logger.info("Quantitative Trading System Started")
    logger.info(f"Trading Pair: {args.symbol} | Period: {args.interval}")
    logger.info("="*60)
    # Run analysis
    if run(args):
        logger.info("Analysis Completed!")
    else:
        logger.error("Analysis Failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
