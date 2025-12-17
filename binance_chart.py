import requests
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import os

def calculate_rsi(series, period=14):
    """
    Calculate RSI indicator
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def fetch_funding_rate(symbol='BTCUSDT'):
    """
    获取币安合约的资金费率
    
    参数:
        symbol: 交易对，如 'BTCUSDT'
    """
    url = 'https://fapi.binance.com/fapi/v1/premiumIndex'
    params = {'symbol': symbol}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        funding_rate = float(data.get('lastFundingRate', 0)) * 100  # 转换为百分比
        next_funding_time = int(data.get('nextFundingTime', 0))
        mark_price = float(data.get('markPrice', 0))
        
        # 转换时间戳
        if next_funding_time > 0:
            next_funding_dt = datetime.fromtimestamp(next_funding_time / 1000)
            next_funding_str = next_funding_dt.strftime('%m-%d %H:%M')
        else:
            next_funding_str = 'N/A'
        
        return {
            'rate': funding_rate,
            'next_time': next_funding_str,
            'mark_price': mark_price
        }
        
    except Exception as e:
        print(f"Failed to fetch funding rate: {e}")
        return None

def fetch_funding_rate_history(symbol='BTCUSDT', limit=30):
    """
    获取历史资金费率数据
    
    参数:
        symbol: 交易对
        limit: 获取的历史记录数量
    """
    url = 'https://fapi.binance.com/fapi/v1/fundingRate'
    params = {
        'symbol': symbol,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        times = []
        rates = []
        
        for item in data:
            timestamp = int(item['fundingTime']) / 1000
            dt = datetime.fromtimestamp(timestamp)
            times.append(dt)
            rates.append(float(item['fundingRate']) * 100)  # 转换为百分比
        
        return times, rates
        
    except Exception as e:
        print(f"Failed to fetch funding rate history: {e}")
        return None, None

def fetch_binance_klines(symbol='BTCUSDT', interval='1h', limit=100, extra_data=0):
    """
    从币安API拉取K线数据
    
    参数:
        symbol: 交易对，如 'BTCUSDT'
        interval: 时间间隔，如 '1m', '5m', '15m', '1h', '4h', '1d'
        limit: 获取的数据条数，最大1000
        extra_data: 额外获取的数据条数，用于计算MA等指标
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit + extra_data
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 转换为DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 转换数据类型并设置索引
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['Open'] = df['open'].astype(float)
        df['High'] = df['high'].astype(float)
        df['Low'] = df['low'].astype(float)
        df['Close'] = df['close'].astype(float)
        df['Volume'] = df['volume'].astype(float)
        df['QuoteVolume'] = df['quote_volume'].astype(float)
        df['Trades'] = df['trades'].astype(int)
        df['TakerBuyBase'] = df['taker_buy_base'].astype(float)
        df['TakerBuyQuote'] = df['taker_buy_quote'].astype(float)
        
        return df
        
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None

def plot_candlestick(df, symbol='BTCUSDT', save_path='binance_chart.png', ma_dict=None, stats=None):
    """
    Use mplfinance to plot candlestick chart with moving averages and statistics
    ma_dict: dictionary with MA period as key and MA series as value
    stats: dictionary with statistical information to display
    """
    if df is None or df.empty:
        print("No data to plot")
        return
    
    import matplotlib.pyplot as plt
    
    # Custom style
    mc = mpf.make_marketcolors(
        up='red', down='green',  # Red up, green down (CN market convention)
        edge='inherit',
        wick='inherit',
        volume='in',
        alpha=0.9
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        gridcolor='#E0E0E0',
        facecolor='white',
        figcolor='white',
        y_on_right=False
    )
    
    # Prepare additional plots for MA lines
    add_plots = []
    ma_colors = ['blue', 'orange', 'purple']
    
    if ma_dict:
        ma_periods = sorted(ma_dict.keys())
        title_ma = '/'.join([f'MA{p}' for p in ma_periods])
        
        for i, period in enumerate(ma_periods):
            add_plots.append(
                mpf.make_addplot(ma_dict[period], color=ma_colors[i % len(ma_colors)], width=1.5)
            )
    else:
        title_ma = ''
    
    title = f'{symbol} Candlestick Chart'
    if title_ma:
        title += f' with {title_ma}'
    
    # Calculate RSI and MACD
    rsi = calculate_rsi(df['Close'])
    macd, signal_line, histogram = calculate_macd(df['Close'])
    
    # Add RSI to additional plots
    add_plots.append(mpf.make_addplot(rsi, panel=2, color='purple', ylabel='RSI', secondary_y=False))
    add_plots.append(mpf.make_addplot([70]*len(df), panel=2, color='red', linestyle='--', width=0.7, alpha=0.5))
    add_plots.append(mpf.make_addplot([30]*len(df), panel=2, color='green', linestyle='--', width=0.7, alpha=0.5))
    
    # Add MACD to additional plots
    add_plots.append(mpf.make_addplot(macd, panel=3, color='blue', ylabel='MACD'))
    add_plots.append(mpf.make_addplot(signal_line, panel=3, color='orange'))
    colors = ['red' if h > 0 else 'green' for h in histogram]
    add_plots.append(mpf.make_addplot(histogram, panel=3, type='bar', color=colors, alpha=0.5))
    
    # Plot using mplfinance with indicators
    fig, axes = mpf.plot(
        df[['Open', 'High', 'Low', 'Close', 'Volume']],
        type='candle',
        style=s,
        title=title,
        ylabel='Price (USDT)',
        volume=True,
        ylabel_lower='Volume',
        addplot=add_plots if add_plots else None,
        figsize=(12, 10),
        returnfig=True,
        panel_ratios=(3, 1, 0.8, 0.8),
        datetime_format='%H:%M',
        xrotation=0
    )
    
    # Simplify x-axis labels for cleaner display
    for ax in fig.get_axes():
        if hasattr(ax, 'xaxis'):
            # Reduce number of x-axis ticks
            ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    
    # Expand figure to make room for summary panel on the right
    if stats:
        # Adjust figure size for indicators
        fig.set_size_inches(13.5, 10)
        
        # Adjust existing axes to make room on the right (maximize chart space)
        for ax in fig.get_axes():
            pos = ax.get_position()
            ax.set_position([pos.x0 * 0.91, pos.y0, pos.width * 0.91, pos.height])
        
        # Add summary panel on the right - positioned higher to avoid y-axis overlap
        ax_summary = fig.add_axes([0.665, 0.38, 0.315, 0.50])
        ax_summary.axis('off')
        
        # Build summary text with technical indicators
        summary_lines = []
        summary_lines.append("══ MARKET ANALYSIS ══\n")
        
        # Price info
        summary_lines.append(f"Price: ${stats['current_price']:,.2f}")
        change_symbol = "↑" if stats['price_change'] >= 0 else "↓"
        summary_lines.append(f"Change: {change_symbol} {abs(stats['price_change_pct']):.2f}%\n")
        
        summary_lines.append(f"Range: ${stats['low']:,.2f}")
        summary_lines.append(f"  ~ ${stats['high']:,.2f}\n")
        
        # Volume and trades
        summary_lines.append(f"Volume: {stats['total_volume']:,.1f}")
        summary_lines.append(f"Trades: {stats['total_trades']:,}\n")
        
        # Market sentiment
        buy_ratio = stats['buy_ratio']
        summary_lines.append("─── SENTIMENT ───")
        summary_lines.append(f"Buy Ratio: {buy_ratio:.1f}%")
        if buy_ratio > 52:
            summary_lines.append("Signal: BULLISH ▲")
        elif buy_ratio < 48:
            summary_lines.append("Signal: BEARISH ▼")
        else:
            summary_lines.append("Signal: NEUTRAL ─")
        summary_lines.append("\n")
        
        # Technical indicators
        if 'rsi' in stats:
            summary_lines.append("─── INDICATORS ───")
            rsi_val = stats['rsi']
            summary_lines.append(f"RSI(14): {rsi_val:.1f}")
            if rsi_val > 70:
                summary_lines.append("  OVERBOUGHT")
            elif rsi_val < 30:
                summary_lines.append("  OVERSOLD")
            else:
                summary_lines.append("  NEUTRAL")
            
            macd_val = stats['macd']
            signal_val = stats['macd_signal']
            summary_lines.append(f"\nMACD: {macd_val:.2f}")
            if macd_val > signal_val:
                summary_lines.append("  BULLISH CROSS")
            else:
                summary_lines.append("  BEARISH CROSS")
            summary_lines.append("\n")
        
        # Funding rate
        if 'funding_rate' in stats and stats['funding_rate'] is not None:
            summary_lines.append("─── FUNDING ───")
            fr = stats['funding_rate']
            summary_lines.append(f"Rate: {fr:+.4f}%")
            summary_lines.append(f"Next: {stats['funding_next']}")
        
        summary_text = '\n'.join(summary_lines)
        
        # Add background box with text
        bbox_props = dict(boxstyle='round,pad=0.8', facecolor='#F8F8F8', edgecolor='#999999', linewidth=2)
        ax_summary.text(0.5, 0.5, summary_text, 
                       transform=ax_summary.transAxes,
                       fontsize=10,
                       verticalalignment='center',
                       horizontalalignment='center',
                       fontfamily='monospace',
                       bbox=bbox_props,
                       linespacing=1.5)
    
    # Add funding rate history chart in the bottom right corner
    if 'funding_history' in stats and stats['funding_history'] is not None:
        times, rates = stats['funding_history']
        if times and rates:
            ax_funding = fig.add_axes([0.78, 0.15, 0.20, 0.18])
            
            # Plot funding rate history
            colors = ['red' if r > 0 else 'green' for r in rates]
            ax_funding.bar(range(len(rates)), rates, color=colors, alpha=0.7, width=0.8)
            ax_funding.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            
            # Styling
            ax_funding.set_title('Funding Rate History', fontsize=9, fontweight='bold', pad=5)
            ax_funding.set_ylabel('%', fontsize=8)
            ax_funding.yaxis.tick_right()
            ax_funding.yaxis.set_label_position('right')
            ax_funding.tick_params(axis='both', labelsize=7)
            ax_funding.grid(True, alpha=0.3, linewidth=0.5)
            
            # X-axis labels (show only a few)
            tick_positions = [0, len(rates)//2, len(rates)-1]
            tick_labels = [times[i].strftime('%m-%d') for i in tick_positions]
            ax_funding.set_xticks(tick_positions)
            ax_funding.set_xticklabels(tick_labels, fontsize=7)
            
            # Add current value annotation
            if rates:
                current_rate = rates[-1]
                ax_funding.text(0.98, 0.95, f'Current: {current_rate:+.4f}%',
                              transform=ax_funding.transAxes,
                              fontsize=7,
                              verticalalignment='top',
                              horizontalalignment='right',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {save_path}")

if __name__ == '__main__':
    # Configuration
    SYMBOL = 'BTCUSDT'  # Trading pair
    INTERVAL = '1h'      # Time interval: 1m, 5m, 15m, 1h, 4h, 1d
    LIMIT = 100          # Number of data points to display
    MA_PERIODS = [7, 25, 99]  # Moving average periods
    
    # Create charts directory if not exists
    os.makedirs('charts', exist_ok=True)
    
    # Fetch extra data for MA calculation (need max MA period - 1 extra points)
    extra_data = max(MA_PERIODS) - 1
    
    print(f"Fetching {SYMBOL} {INTERVAL} kline data from Binance...")
    df = fetch_binance_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, extra_data=extra_data)
    
    # Fetch funding rate and history
    print(f"Fetching funding rate for {SYMBOL}...")
    funding_info = fetch_funding_rate(symbol=SYMBOL)
    funding_times, funding_rates = fetch_funding_rate_history(symbol=SYMBOL, limit=30)
    
    if df is not None:
        print(f"Successfully fetched {len(df)} data points (including {extra_data} extra for MA calculation)")
        
        # Calculate MA on full dataset first
        ma_dict = {}
        for period in MA_PERIODS:
            ma_dict[period] = df['Close'].rolling(window=period).mean()
        
        # Truncate to display only the last LIMIT points
        df_display = df.iloc[-LIMIT:].copy()
        ma_dict_display = {}
        for period, ma_series in ma_dict.items():
            ma_dict_display[period] = ma_series.iloc[-LIMIT:]
        
        # Calculate statistics
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
        
        # Calculate technical indicators for summary
        rsi_full = calculate_rsi(df['Close'])
        macd_full, signal_full, _ = calculate_macd(df['Close'])
        current_rsi = rsi_full.iloc[-1]
        current_macd = macd_full.iloc[-1]
        current_signal = signal_full.iloc[-1]
        
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
            'macd_signal': current_signal
        }
        
        # Add funding rate info to stats
        if funding_info:
            stats['funding_rate'] = funding_info['rate']
            stats['funding_next'] = funding_info['next_time']
        
        # Add funding rate history
        if funding_times and funding_rates:
            stats['funding_history'] = (funding_times, funding_rates)
        
        print(f"Displaying last {len(df_display)} data points")
        print(f"\nLatest price: {current_price:.2f}")
        print(f"Price change: {price_change:+.2f} ({price_change_pct:+.2f}%)")
        print(f"High: {high_price:.2f} | Low: {low_price:.2f}")
        print(f"Total volume: {total_volume:.2f}")
        print(f"Total trades: {total_trades:,}")
        print(f"Buy ratio: {buy_ratio:.1f}%")
        
        if funding_info:
            print(f"Funding rate: {funding_info['rate']:+.4f}% (Next: {funding_info['next_time']})")
            print(f"Mark price: {funding_info['mark_price']:.2f}")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'charts/{SYMBOL}_{INTERVAL}_{timestamp}.png'
        
        # Plot candlestick chart with MA lines and statistics
        plot_candlestick(df_display, symbol=SYMBOL, save_path=save_path, ma_dict=ma_dict_display, stats=stats)
        
        # Optional: Analyze chart with AI
        # Uncomment the following lines to enable automatic analysis
        """
        print("\nAnalyzing chart with AI...")
        from chart_analyzer_api import ChartAnalyzerAPI
        # 使用通义千问: service='qwen'
        # 使用智谱GLM-4V: service='glm'
        # 使用文心一言: service='wenxin'
        analyzer = ChartAnalyzerAPI(service='qwen')
        analyzer.analyze_chart_and_save(save_path, save_json=True)
        print("Analysis completed!")
        """
