#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Fear & Greed Index Fetcher - 获取加密货币恐慌贪婪指数

Alternative.me的Fear & Greed Index是一个综合指标，基于以下因素：
- 波动性 (25%)
- 市场动量/交易量 (25%)
- 社交媒体 (15%)
- 调查 (15%)
- 比特币市场份额 (10%)
- Google趋势 (10%)

指数范围：0-100
- 0-24: 极度恐慌 (Extreme Fear)
- 25-49: 恐慌 (Fear)
- 50: 中性 (Neutral)
- 51-75: 贪婪 (Greed)
- 76-100: 极度贪婪 (Extreme Greed)
"""

import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class FearGreedFetcher:
    """Fear & Greed Index数据获取器"""
    
    API_URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        """初始化"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_current_index(self) -> Dict[str, Any]:
        """
        获取当前的Fear & Greed Index
        
        Returns:
            字典包含：
            - value: 指数值 (0-100)
            - classification: 分类 (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)
            - sentiment_score: 情绪得分 (0-1)
            - timestamp: 时间戳
        """
        try:
            response = self.session.get(
                self.API_URL,
                params={"limit": 1},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if 'data' not in data or len(data['data']) == 0:
                logger.error("API返回数据格式错误")
                return self._default_response()
            
            latest = data['data'][0]
            value = int(latest['value'])
            classification = latest['value_classification']
            timestamp = int(latest['timestamp'])
            
            # 转换为0-1的情绪得分
            sentiment_score = value / 100
            
            # logger.info(f"Fear & Greed Index: {value} ({classification})")
            
            return {
                'value': value,
                'classification': classification,
                'sentiment_score': sentiment_score,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'confidence': 0.85,  # 这个指标综合了多个数据源，置信度较高
            }
            
        except Exception as e:
            logger.error(f"获取Fear & Greed Index失败: {e}", exc_info=True)
            return self._default_response()
    
    def get_historical(self, limit: int = 30) -> pd.DataFrame:
        """
        获取历史Fear & Greed Index数据
        
        Args:
            limit: 获取的数据点数量（最多365）
            
        Returns:
            DataFrame包含历史数据
        """
        try:
            response = self.session.get(
                self.API_URL,
                params={"limit": min(limit, 365)},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if 'data' not in data:
                logger.error("API返回数据格式错误")
                return pd.DataFrame()
            
            records = []
            for item in data['data']:
                records.append({
                    'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                    'value': int(item['value']),
                    'classification': item['value_classification'],
                    'sentiment_score': int(item['value']) / 100
                })
            
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # logger.info(f"获取到 {len(df)} 天的历史数据")
            return df
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_trend(self, days: int = 7) -> Dict[str, Any]:
        """
        分析指数趋势
        
        Args:
            days: 分析的天数
            
        Returns:
            趋势分析结果
        """
        df = self.get_historical(limit=days)
        
        if df.empty or len(df) < 2:
            return {
                'trend': 'unknown',
                'change': 0,
                'avg_value': 50
            }
        
        current_value = df['value'].iloc[-1]
        previous_value = df['value'].iloc[0]
        avg_value = df['value'].mean()
        change = current_value - previous_value
        
        if change > 5:
            trend = 'increasing'  # 贪婪度上升
        elif change < -5:
            trend = 'decreasing'  # 恐慌度上升
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': change,
            'current': current_value,
            'avg_value': avg_value,
            'days': days
        }
    
    def interpret_signal(self, index_value: int) -> Dict[str, Any]:
        """
        解读交易信号
        
        传统反向指标策略：
        - 极度恐慌(0-24)时买入（市场超卖）
        - 极度贪婪(76-100)时卖出（市场超买）
        
        Args:
            index_value: 指数值 (0-100)
            
        Returns:
            交易信号解读
        """
        if index_value <= 24:
            signal = "strong_buy"
            description = "极度恐慌 - 考虑买入（反向指标）"
            position = 0.8  # 做多
        elif index_value <= 49:
            signal = "buy"
            description = "恐慌 - 可以逢低买入"
            position = 0.4
        elif index_value <= 55:
            signal = "neutral"
            description = "中性 - 观望"
            position = 0.0
        elif index_value <= 75:
            signal = "sell"
            description = "贪婪 - 考虑获利了结"
            position = -0.4
        else:
            signal = "strong_sell"
            description = "极度贪婪 - 考虑卖出（反向指标）"
            position = -0.8
        
        return {
            'signal': signal,
            'description': description,
            'position': position,
            'reasoning': f"Fear & Greed Index: {index_value}/100 - {description}"
        }
    
    def _default_response(self) -> Dict[str, Any]:
        """默认响应（当API失败时）"""
        return {
            'value': 50,
            'classification': 'Neutral',
            'sentiment_score': 0.5,
            'timestamp': int(datetime.now().timestamp()),
            'datetime': datetime.now().isoformat(),
            'confidence': 0.0,
        }


def main():
    """测试函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    fetcher = FearGreedFetcher()
    
    # 获取当前指数
    print("\n" + "="*60)
    print("Fear & Greed Index - 当前状态")
    print("="*60)
    
    current = fetcher.get_current_index()
    print(f"\n指数值: {current['value']}/100")
    print(f"分类: {current['classification']}")
    print(f"情绪得分: {current['sentiment_score']:.2f}")
    print(f"更新时间: {current['datetime']}")
    
    # 获取交易信号
    signal = fetcher.interpret_signal(current['value'])
    print(f"\n交易信号: {signal['signal']}")
    print(f"建议仓位: {signal['position']:.2f}")
    print(f"说明: {signal['description']}")
    
    # 获取趋势
    print("\n" + "="*60)
    print("7天趋势分析")
    print("="*60)
    
    trend = fetcher.get_trend(days=7)
    print(f"\n趋势: {trend['trend']}")
    print(f"变化: {trend['change']:+.0f}")
    print(f"当前值: {trend['current']}")
    print(f"7日均值: {trend['avg_value']:.1f}")
    
    # 获取历史数据
    print("\n" + "="*60)
    print("30天历史数据")
    print("="*60)
    
    df = fetcher.get_historical(limit=30)
    if not df.empty:
        print(f"\n获取到 {len(df)} 天的数据")
        print(f"最高: {df['value'].max()}")
        print(f"最低: {df['value'].min()}")
        print(f"平均: {df['value'].mean():.1f}")
        print(f"\n最近5天:")
        print(df.tail()[['value', 'classification']])


if __name__ == "__main__":
    main()
