#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import sys
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


def run(symbol='BTCUSDT', interval='1h', limit=100, 
                         ma_periods=[7, 25, 99], service='copilot', model=None):
    if service == 'copilot' and model:
        print(f"AI服务: GitHub Copilot ({model})")
    else:
        print(f"AI服务: {service.upper()}")
    print()
    # 创建图表目录
    os.makedirs('charts', exist_ok=True)
    # 1. 获取K线数据
    print("📈 步骤 1/4: 获取K线数据...")
    extra_data = max(ma_periods) - 1
    df = fetch_binance_klines(
        symbol=symbol, interval=interval, 
        limit=limit, extra_data=extra_data)
    if df is None:
        print("❌ 获取数据失败")
        return None
    print(f"✓ 成功获取 {len(df)} 条数据")
    # 2. 获取资金费率
    print("\n💰 步骤 2/4: 获取资金费率...")
    funding_info = fetch_funding_rate(symbol=symbol)
    funding_times, funding_rates = fetch_funding_rate_history(symbol=symbol, limit=30)
    if funding_info:
        print(f"✓ 当前资金费率: {funding_info['rate']:+.4f}%")
    # 3. 计算指标并生成图表
    print("\n📊 步骤 3/4: 计算技术指标并生成图表...")
    # 计算均线
    ma_dict = {}
    for period in ma_periods:
        ma_dict[period] = df['Close'].rolling(window=period).mean()
    # 只显示最后limit条数据
    df_display = df.iloc[-limit:].copy()
    ma_dict_display = {}
    for period, ma_series in ma_dict.items():
        ma_dict_display[period] = ma_series.iloc[-limit:]
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
    # 技术指标
    rsi_full = calculate_rsi(df['Close'])
    macd_full, signal_full, _ = calculate_macd(df['Close'])
    current_rsi = rsi_full.iloc[-1]
    current_macd = macd_full.iloc[-1]
    current_signal = signal_full.iloc[-1]
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
        'atr_pct': atr_pct
    }
    if funding_info:
        stats['funding_rate'] = funding_info['rate']
        stats['funding_next'] = funding_info['next_time']
    if funding_times and funding_rates:
        stats['funding_history'] = (funding_times, funding_rates)
    # 生成图表文件名
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'charts/{symbol}_{interval}_{timestamp}.png'
    # 绘制图表
    plot_candlestick(
        df_display, symbol=symbol, save_path=save_path, 
        ma_dict=ma_dict_display, stats=stats)
    print(f"✓ 图表已保存: {save_path}")
    # 4. AI分析
    model_display = f"{service.upper()}"
    if service == 'copilot' and model:
        model_display = f"GitHub Copilot ({model})"
    print(f"\n🤖 步骤 4/4: 使用 {model_display} 进行AI分析...")
    try:
        advisor = PositionAdvisor(service=service, model=model)
        result = advisor.analyze(save_path, save_json=True)
        return {
            'chart_path': save_path,
            'stats': stats,
            'analysis': result
        }
    except Exception as e:
        print(f"❌ AI分析失败: {e}")
        print(f"\n💡 提示: 请确保设置了正确的API密钥环境变量")
        print(f"   - GitHub Copilot: GITHUB_TOKEN")
        print(f"   - OpenAI: OPENAI_API_KEY")
        print(f"   - 通义千问: DASHSCOPE_API_KEY")
        print(f"   - DeepSeek: DEEPSEEK_API_KEY")
        return {
            'chart_path': save_path,
            'stats': stats,
            'analysis': None
        }


def main():
    """命令行入口"""
    load_dotenv()
    # 解析参数
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT'
    interval = sys.argv[2] if len(sys.argv) > 2 else '1h'
    service = sys.argv[3] if len(sys.argv) > 3 else 'copilot'
    model = sys.argv[4] if len(sys.argv) > 4 else None
    
    # 运行分析
    result = run(symbol=symbol, interval=interval, service=service, model=model)
    if result:
        print("\n✅ 分析完成！")
    else:
        print("\n❌ 分析失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
