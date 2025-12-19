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
    plot_candlestick,
    calculate_rsi,
    calculate_macd,
)
from vquant.analysis.advisor import PositionAdvisor
from vquant.analysis.predictor import QuantPredictor
from vquant.analysis.wave import WaveTrader


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
    logger.info("Step 3/4: Calculating technical indicators and generating chart...")
    # 计算均线
    ma_dict = {}
    for period in args.ma_periods:
        ma_dict[period] = df["Close"].rolling(window=period).mean()
    # 只显示最后limit条数据
    df_display = df.iloc[-args.limit :].copy()
    ma_dict_display = {}
    for period, ma_series in ma_dict.items():
        ma_dict_display[period] = ma_series.iloc[-args.limit :]
    # 计算统计数据
    current_price = df_display.iloc[-1]["Close"]
    first_price = df_display.iloc[0]["Open"]
    price_change = current_price - first_price
    price_change_pct = (price_change / first_price) * 100
    high_price = df_display["High"].max()
    low_price = df_display["Low"].min()
    total_volume = df_display["Volume"].sum()
    total_trades = df_display["Trades"].sum()
    total_taker_buy = df_display["TakerBuyBase"].sum()
    buy_ratio = (total_taker_buy / total_volume * 100) if total_volume > 0 else 0
    # 技术指标 - 在完整数据上计算，然后截断用于显示
    rsi_full = calculate_rsi(df["Close"])
    macd_full, signal_full, histogram_full = calculate_macd(df["Close"])
    current_rsi = rsi_full.iloc[-1]
    current_macd = macd_full.iloc[-1]
    current_signal = signal_full.iloc[-1]
    # 截断指标数据用于绘图
    rsi_display = rsi_full.iloc[-args.limit :]
    macd_display = macd_full.iloc[-args.limit :]
    signal_display = signal_full.iloc[-args.limit :]
    histogram_display = histogram_full.iloc[-args.limit :]
    # 市场动态指标
    volatility = df_display["Close"].pct_change().std() * 100
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
    if len(df_display) >= 50:
        # Current momentum (already calculated above)
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
    
    recent_volume = df_display.iloc[-10:]["Volume"].mean()
    avg_volume = df_display["Volume"].mean()
    volume_strength = (
        ((recent_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
    )
    # ATR
    high_low = df_display["High"] - df_display["Low"]
    high_close = abs(df_display["High"] - df_display["Close"].shift())
    low_close = abs(df_display["Low"] - df_display["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().iloc[-1]
    atr_pct = (atr / current_price * 100) if current_price > 0 else 0
    stats = {
        "current_price": current_price,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "high": high_price,
        "low": low_price,
        "total_volume": total_volume,
        "total_trades": total_trades,
        "buy_ratio": buy_ratio,
        "rsi": current_rsi,
        "current_rsi": current_rsi,  # For predictor
        "macd": current_macd,
        "current_macd": current_macd,  # For predictor
        "macd_signal": current_signal,
        "current_signal": current_signal,  # For predictor
        "volatility": volatility,
        "momentum": momentum,
        "impulse": impulse,
        "volume_strength": volume_strength,
        "atr": atr,
        "atr_pct": atr_pct,
        # 添加完整的指标序列用于绘图
        "rsi_series": rsi_display,
        "macd_series": macd_display,
        "signal_series": signal_display,
        "histogram_series": histogram_display,
        # 添加均线数据和拐点检测
        "ma_dict": {
            period: ma_series.iloc[-1] for period, ma_series in ma_dict.items()
        },
        "ma_periods": args.ma_periods,
        # 添加最近的K线数据用于表格展示
        "recent_klines": df_display.iloc[-24:]
        .reset_index()[["timestamp", "Open", "High", "Low", "Close"]]
        .to_dict("records"),
        "recent_ma": {
            period: ma_series.iloc[-24:].tolist()
            for period, ma_series in ma_dict.items()
        },
    }
    
    # 添加当前MA值供predictor使用
    if 7 in ma_dict:
        stats["current_ma7"] = ma_dict[7].iloc[-1]
    if 25 in ma_dict:
        stats["current_ma25"] = ma_dict[25].iloc[-1]
    if 99 in ma_dict:
        stats["current_ma99"] = ma_dict[99].iloc[-1]
    # 计算MA7拐点（检测最近4根K线内的拐点）
    if 7 in ma_dict and len(ma_dict[7]) >= 5:
        ma7_series = ma_dict[7]

        # 检查最近4根K线内是否有拐点
        inflection_found = None
        for i in range(1, 5):  # 检查最近4根bar
            if len(ma7_series) >= i + 2:
                ma7_current = ma7_series.iloc[-i]
                ma7_prev1 = ma7_series.iloc[-i - 1]
                ma7_prev2 = ma7_series.iloc[-i - 2]

                slope_recent = ma7_current - ma7_prev1
                slope_before = ma7_prev1 - ma7_prev2

                if slope_before > 0 and slope_recent < 0:
                    inflection_found = "downward"
                    break
                elif slope_before < 0 and slope_recent > 0:
                    inflection_found = "upward"
                    break

        stats["ma7_inflection"] = inflection_found if inflection_found else "continuing"
    if funding_info:
        stats["funding_rate"] = funding_info["rate"]
        stats["funding_next"] = funding_info["next_time"]
    if funding_times and funding_rates:
        stats["funding_history"] = (funding_times, funding_rates)
    
    # 生成图表 - 仅在使用LLM分析时需要
    save_path = None
    image_bytes = None
    
    if args.predictor == "llm":
        # LLM需要图表进行视觉分析
        if args.quiet:
            logger.info("Quiet mode: generating chart in memory (no file I/O)")
            image_bytes = plot_candlestick(
                df_display,
                symbol=args.symbol,
                save_path=None,
                ma_dict=ma_dict_display,
                stats=stats,
                return_bytes=True,
            )
        else:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"charts/{args.name}_{timestamp}.png"
            plot_candlestick(
                df_display,
                symbol=args.symbol,
                save_path=save_path,
                ma_dict=ma_dict_display,
                stats=stats,
                return_bytes=False,
            )
            logger.info(f"Chart saved: {save_path}")
    else:
        logger.info("Quantitative predictor mode: skipping chart generation")
    
    # 4. Analysis - Use Predictor, Wave Trader, or AI Advisor
    result = None
    
    if args.predictor == "wave":
        # Use wave trading strategy
        logger.info("Step 4/4: Running wave trader...")
        try:
            trader = WaveTrader(
                symbol=args.symbol,
                buy_threshold=args.wave_buy_threshold,
                sell_threshold=args.wave_sell_threshold,
                min_trade_interval=args.wave_interval,
                max_position=args.volume if args.volume > 0 else 1.0,
                state_file=f"data/wave_state_{args.name}.json"
            )
            
            # Generate trading signal
            signal = trader.generate_signal(
                current_price=current_price,
                volume=args.volume if args.volume > 0 else 0.1,
                stats=stats
            )
            
            # Convert signal to standard result format
            result = {
                'symbol': args.symbol,
                'position': 1.0 if signal['action'] == 'buy' else (-1.0 if signal['action'] == 'sell' else 0.0),
                'confidence': 'high' if signal['action'] != 'hold' else 'low',
                'current_price': current_price,
                'reasoning': signal['reasoning'],
                'analysis_type': 'wave',
                'trade_action': signal['action'],
                'trade_volume': signal['volume'],
                'price_change': signal.get('price_change'),
                'state_summary': trader.get_state_summary()
            }
            
            logger.info(f"Wave signal: {signal['action'].upper()} - {signal['reasoning']}")
            
        except Exception as e:
            logger.exception(f"Wave trading failed: {e}", exc_info=True)
            return False
    
    elif args.predictor == "quant":
        # 使用量化预测模型
        logger.info("Step 4/4: Running quantitative predictor...")
        try:
            predictor = QuantPredictor(symbol=args.symbol)
            pred_result = predictor.predict(stats)
            
            # 构建标准化输出
            result = {
                'symbol': args.symbol,
                'position': pred_result['position'],
                'confidence': pred_result['confidence'],
                'current_price': current_price,
                'score': pred_result.get('score'),
                'factors': pred_result.get('factors'),
                'reasoning': pred_result['reasoning'],
                'analysis_type': 'predictor',
            }
            
            logger.info(f"Prediction result: position={result['position']}, confidence={result['confidence']}")
            logger.info(f"Reasoning: {result['reasoning']}")
            
        except Exception as e:
            logger.exception(f"Quantitative prediction failed: {e}", exc_info=True)
            return False
    else:
        # 使用AI Advisor（原有逻辑）
        logger.info(
            f"Step 4/4: Performing AI analysis with {args.service}::{args.model}..."
        )
        try:
            advisor = PositionAdvisor(service=args.service, model=args.model)
            result = advisor.analyze(
                image_path=save_path,
                image_bytes=image_bytes,
                symbol=args.symbol,
                interval=args.interval,
                current_price=current_price,
                stats=stats,
            )
            result['analysis_type'] = 'advisor'
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}", exc_info=True)
            logger.info(
                "Tip: Please ensure you have set the correct API key environment variables"
            )
            logger.info("  - GitHub Copilot: GITHUB_TOKEN")
            logger.info("  - OpenAI: OPENAI_API_KEY")
            logger.info("  - Qwen: DASHSCOPE_API_KEY")
            logger.info("  - DeepSeek: DEEPSEEK_API_KEY")
            return False
    
    # Save result to JSON
    if result and not args.quiet:
        import json
        if save_path:
            # 如果有图表路径，保存在图表旁边
            result_json_path = save_path.replace('.png', '.json')
        else:
            # 量化预测器模式，单独保存JSON
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
        choices=["llm", "quant", "wave"],
        help="Analysis method: 'llm' for AI advisor (default), 'quant' for quantitative predictor, 'wave' for wave trader",
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
    # Wave trader parameters
    parser.add_argument(
        "--wave-buy-threshold",
        type=float,
        default=-0.5,
        help="Wave trader: price drop percentage to trigger buy (default: -0.5)",
    )
    parser.add_argument(
        "--wave-sell-threshold",
        type=float,
        default=0.5,
        help="Wave trader: price rise percentage to trigger sell (default: 0.5)",
    )
    parser.add_argument(
        "--wave-interval",
        type=int,
        default=300,
        help="Wave trader: minimum seconds between trades (default: 300)",
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
