#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import sys
import logging
import argparse
import pandas as pd

from dotenv import load_dotenv
from vquant.model.vision import (
    fetch_binance_klines,
    fetch_funding_rate,
    fetch_funding_rate_history,
)
from vquant.data.indicators import prepare_market_stats, add_funding_rate_to_stats
from vquant.analysis.advisor import PositionAdvisor
from vquant.analysis.quant import QuantPredictor
from vquant.analysis.martin import MartinTrader
from vquant.analysis.kelly import KellyTrader
from vquant.analysis.kalshi import KalshiTrader


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
    log_format = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    if args.log_file == "auto":
        args.log_file = f"logs/{args.name}.log"
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        handlers = [logging.FileHandler(args.log_file, encoding="utf-8")]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=level, format=log_format, datefmt=date_format, handlers=handlers
    )


def run(args):
    # main logical
    # Create charts directory
    os.makedirs("charts", exist_ok=True)
    # 1. Fetch K-line data
    logger.info("Step 1/4: Fetching K-line data...")
    extra_data = max(args.ma_periods) - 1
    df = fetch_binance_klines(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        extra_data=extra_data,
    )
    if df is None:
        logger.error("Failed to fetch data")
        return None
    logger.info(f"Successfully fetched {len(df)} data points")
    
    # 2. Fetch funding rate
    logger.info("Step 2/4: Fetching funding rate...")
    funding_info = fetch_funding_rate(symbol=args.symbol)
    funding_times, funding_rates = fetch_funding_rate_history(
        symbol=args.symbol, limit=30
    )
    if funding_info:
        logger.info(f"Current funding rate: {funding_info['rate']:.4f}%")
    
    # 3. Calculate indicators and generate chart
    logger.info("Step 3/4: Calculating technical indicators...")
    
    # Calculate moving averages
    ma_dict = {}
    for period in args.ma_periods:
        ma_dict[period] = df["Close"].rolling(window=period).mean()
    
    # Display data (last limit rows)
    df_display = df.iloc[-args.limit :].copy()
    ma_dict_display = {}
    for period, ma_series in ma_dict.items():
        ma_dict_display[period] = ma_series.iloc[-args.limit :]
    
    # Prepare comprehensive market statistics using shared module
    stats = prepare_market_stats(df, df_display, ma_dict, args)
    
    # Add funding rate data to stats
    add_funding_rate_to_stats(stats, funding_info, funding_times, funding_rates)
    
    # 4. Analysis - Use unified predictor interface
    logger.info(f"Step 4/4: Running {args.predictor} analysis...")
    
    try:
        # Create appropriate predictor instance
        predictor = _create_predictor(args)
        
        # Run analysis using unified interface
        result, save_path = predictor.run(
            df=df,
            df_display=df_display,
            ma_dict=ma_dict,
            ma_dict_display=ma_dict_display,
            stats=stats,
            args=args
        )
        
        # Log result
        logger.info(f"Analysis result: position={result.get('position')}, confidence={result.get('confidence')}")
        logger.info(f"Reasoning: {result.get('reasoning', 'N/A')}")
        
    except Exception as e:
        logger.exception(f"Analysis failed: {e}", exc_info=True)
        if args.predictor == "llm":
            logger.info("Tip: Please ensure you have set the correct API key environment variables")
            logger.info("  - GitHub Copilot: GITHUB_TOKEN")
            logger.info("  - OpenAI: OPENAI_API_KEY")
            logger.info("  - Qwen: DASHSCOPE_API_KEY")
            logger.info("  - DeepSeek: DEEPSEEK_API_KEY")
        return False
    
    # Save result to JSON
    if result and not args.quiet:
        import json
        if save_path:
            # If chart path exists, save JSON next to it
            result_json_path = save_path.replace('.png', '.json')
        else:
            # No chart path, save JSON independently
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            result_json_path = f"charts/{args.name}_{timestamp}.json"
        
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Result saved: {result_json_path}")
    
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
            logger.exception(
                f"Trade execution failed: {trade_error}", exc_info=True
            )
    return True


def _create_predictor(args):
    """Create appropriate predictor instance based on args"""
    if args.predictor == "quant":
        return QuantPredictor(
            symbol=args.symbol,
            name=args.name,
            config_dir="config"
        )
    elif args.predictor == "martin":
        return MartinTrader(
            symbol=args.symbol,
            name=args.name,
            config_dir="config"
        )
    elif args.predictor == "kelly":
        return KellyTrader(
            symbol=args.symbol,
            name=args.name,
            config_dir="config"
        )
    elif args.predictor == "kalshi":
        return KalshiTrader(
            symbol=args.symbol,
            name=args.name,
            email=args.kalshi_email,
            password=args.kalshi_password,
            config_path=args.kalshi_config
        )
    else:  # llm
        return PositionAdvisor(
            symbol=args.symbol,
            name=args.name,
            service=args.service,
            model=args.model
        )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Quantitative Trading System - K-line Analysis and Auto Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    # Basic parameters
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDC",
        help="Trading pair symbol (default: BTCUSDC)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        help="K-line period (default: 1h)",
    )
    # Analysis method selection
    parser.add_argument(
        "--predictor",
        "-p",
        type=str,
        default="llm",
        choices=["llm", "quant", "martin", "kelly", "kalshi"],
        help="Analysis method: 'llm' for AI advisor (default), 'quant' for quantitative predictor, "
             "'martin' for martingale trader, 'kelly' for Kelly Criterion trader, "
             "'kalshi' for Kalshi prediction market based trader",
    )
    # AI service configuration
    parser.add_argument(
        "--service",
        "-s",
        type=str,
        default="qwen",
        choices=["copilot", "openai", "qwen", "deepseek"],
        help="AI service provider (default: qwen, ignored if --use-predictor is set)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="AI model name (copilot options: gpt-4o, claude-3.5-sonnet, o1-preview, etc.)",
    )
    # Kalshi strategy parameters
    parser.add_argument(
        "--kalshi-email",
        type=str,
        help="Kalshi account email (optional, for accessing private data)",
    )
    parser.add_argument(
        "--kalshi-password",
        type=str,
        help="Kalshi account password (optional)",
    )
    parser.add_argument(
        "--kalshi-config",
        type=str,
        default="config/kalshi_strategy.json",
        help="Kalshi strategy configuration file path (default: config/kalshi_strategy.json)",
    )
    # Trading parameters
    parser.add_argument(
        "--trade",
        "-t",
        action="store_true",
        help="Enable trading execution mode (default: analysis only)",
    )
    parser.add_argument(
        "--volume", type=float, default=0, help="max trading volumes(default: 0.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,  # (0,1)
        help="using with volume, > threshold then xxx < -theshold then xxx (default: 0.5)",
    )
    parser.add_argument(
        "--account",
        "-a",
        type=str,
        default="li",
        help="Trading account name (default: li)",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Strategy name (must be unique, prevents duplicate instances) (default: default)",
    )
    # Technical parameters
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=72,
        help="Number of K-line data points (default: 72)",
    )
    parser.add_argument(
        "--ma-periods",
        type=int,
        nargs="+",
        default=[7, 25, 99],
        help="Moving average periods (default: 7 25 99)",
    )
    # Logging configuration
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging output"
    )
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode, only output errors"
    )
    return parser.parse_args()


def main():
    """Command line entry point"""
    # Load environment variables
    load_dotenv()
    # Parse command line arguments
    args = parse_arguments()
    # Initialize logging system
    setup_logging(args)
    logger.info("=" * 60)
    logger.info("Quantitative Trading System Started")
    logger.info(f"Trading Pair: {args.symbol} | Period: {args.interval}")
    logger.info("=" * 60)
    # Run analysis
    if run(args):
        logger.info("Analysis Completed!")
    else:
        logger.error("Analysis Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
