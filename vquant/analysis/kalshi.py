#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kalshi Trader - 基于Kalshi预测市场数据的数字货币交易策略

策略逻辑：
1. 从Kalshi获取加密货币相关的预测市场数据
2. 计算市场情绪指标和预测概率
3. 结合技术指标做出交易决策
4. 动态调整仓位

核心思想：
- 利用预测市场的"群体智慧"来预测价格走势
- 预测市场价格反映了参与者对未来事件的集体预期
- 高置信度的市场预测可以作为交易信号
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

from .base import BasePredictor
from vquant.data.kalshi_fetcher import KalshiFetcher
from vquant.data.fear_greed_fetcher import FearGreedFetcher
from vquant.model.vision import fetch_binance_klines


logger = logging.getLogger(__name__)


class KalshiTrader(BasePredictor):
    """基于预测市场的交易策略
    
    数据源优先级：
    1. Fear & Greed Index (主要数据源)
    2. Kalshi预测市场 (如果有加密货币市场)
    """
    
    def __init__(self, 
                 symbol: str = "BTCUSDC",
                 name: str = "kalshi",
                 email: Optional[str] = None,
                 password: Optional[str] = None,
                 config_path: Optional[str] = None,
                 use_fear_greed: bool = True):
        """
        初始化交易策略
        
        Args:
            symbol: 交易对符号
            name: 策略名称
            email: Kalshi账户邮箱（可选）
            password: Kalshi账户密码（可选）
            config_path: 配置文件路径
            use_fear_greed: 是否使用Fear & Greed Index
        """
        super().__init__(symbol, name)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化数据获取器
        self.use_fear_greed = use_fear_greed
        if use_fear_greed:
            self.fear_greed_fetcher = FearGreedFetcher()
            logger.info("使用Fear & Greed Index作为主要数据源")
        
        self.kalshi_fetcher = KalshiFetcher(email, password) if email and password else KalshiFetcher()
        
        # 策略参数
        self.sentiment_threshold_long = self.config.get("sentiment_threshold_long", 0.65)
        self.sentiment_threshold_short = self.config.get("sentiment_threshold_short", 0.35)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)
        self.max_position = self.config.get("max_position", 1.0)
        self.use_technical_filter = self.config.get("use_technical_filter", True)
        
        # Fear & Greed特定参数
        self.fear_greed_mode = self.config.get("fear_greed_mode", "contrarian")  # contrarian或momentum
        
        # 仓位稳定性参数
        self.position_change_threshold = self.config.get("position_change_threshold", 0.15)  # 最小仓位变化阈值
        self.min_hold_minutes = self.config.get("min_hold_minutes", 60)  # 最小持仓时间（分钟）
        
        # 仓位状态记录
        self.last_position = 0.0
        self.last_position_time = None
        self.position_history = []
        
        # K线数据缓存
        self.cached_df = None
        self.cached_sentiment_data = None  # 缓存sentiment数据
    
    def predict(self, df: pd.DataFrame) -> float:
        """
        预测仓位
        
        Args:
            df: K线数据
            
        Returns:
            建议仓位 (-1到1之间)
        """
        try:
            # 1. 获取市场情绪
            sentiment_data = self._get_market_sentiment()
            sentiment_score = sentiment_data['sentiment_score']
            confidence = sentiment_data['confidence']
            
            # 2. 检查置信度
            if confidence < self.confidence_threshold:
                logger.warning(f"置信度过低({confidence:.3f} < {self.confidence_threshold})，建议空仓")
                raw_position = 0.0
            else:
                # 3. 基于情绪得分计算基础仓位
                raw_position = 0.0
                
                if sentiment_score >= self.sentiment_threshold_long:
                    # 做多信号
                    raw_position = (sentiment_score - 0.5) * 2  # 映射到0-1
                    raw_position = min(raw_position, self.max_position)
                    
                elif sentiment_score <= self.sentiment_threshold_short:
                    # 做空信号
                    raw_position = (sentiment_score - 0.5) * 2  # 映射到-1-0
                    raw_position = max(raw_position, -self.max_position)
                
                else:
                    raw_position = 0.0
                
                # 4. 技术指标过滤
                if self.use_technical_filter:
                    indicators = self._calculate_technical_indicators(df)
                    raw_position = self._apply_technical_filter(raw_position, indicators)
                
                # 5. 置信度调整
                raw_position *= confidence
                
                # 6. 限制在最大仓位范围内
                raw_position = np.clip(raw_position, -self.max_position, self.max_position)
            
            # 7. 应用仓位稳定性机制
            stable_position = self._apply_position_stability(raw_position)
            
            return stable_position
            
        except Exception as e:
            logger.error(f"预测失败: {e}", exc_info=True)
            return 0.0
    
    def analyze(self, df: pd.DataFrame = None, interval: str = "1h", days: int = 7, stats: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        分析市场并给出交易建议
        
        Args:
            df: K线数据DataFrame（优先使用）
            interval: K线间隔
            days: 回看天数
            stats: 统计数据
            **kwargs: 其他参数
            
        Returns:
            分析结果字典
        """
        try:
            # 优先使用cached_df（来自prepare_data）
            if df is None and self.cached_df is not None:
                df = self.cached_df
                logger.info(f"使用缓存的K线数据：{len(df)}条")
            
            # 如果仍然没有df，尝试获取
            if df is None or df.empty:
                logger.warning("未提供K线数据，尝试自行获取...")
                df = fetch_binance_klines(self.symbol, interval, days)
                if df.empty:
                    logger.error("获取K线数据失败")
                    return {"position": 0.0, "error": "数据获取失败"}
            
            # 预测仓位
            position = self.predict(df)
            
            # 获取情绪数据
            sentiment_data = self._get_market_sentiment()
            
            # 计算技术指标
            indicators = self._calculate_technical_indicators(df)
            
            # 生成分析报告
            result = {
                "symbol": self.symbol,
                "strategy": "kalshi",
                "timestamp": datetime.now().isoformat(),
                "position": position,
                "sentiment": sentiment_data,
                "technical_indicators": indicators,
                "action": self._get_action_description(position),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分析失败: {e}", exc_info=True)
            return {"position": 0.0, "error": str(e)}
    
    def _get_action_description(self, position: float) -> str:
        """获取操作描述"""
        if position > 0.7:
            return "强烈建议做多"
        elif position > 0.3:
            return "建议做多"
        elif position > 0.1:
            return "小仓位做多"
        elif position < -0.7:
            return "强烈建议做空"
        elif position < -0.3:
            return "建议做空"
        elif position < -0.1:
            return "小仓位做空"
        else:
            return "建议空仓观望"
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = "config/kalshi_strategy.json"
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"已加载配置文件: {config_path}")
                return config
        else:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return {}
    
    def _get_market_sentiment(self) -> Dict[str, Any]:
        """获取市场情绪数据（带缓存）"""
        # 如果有缓存，直接返回
        if self.cached_sentiment_data is not None:
            return self.cached_sentiment_data
        
        # 获取新数据并缓存
        if self.use_fear_greed:
            self.cached_sentiment_data = self._get_fear_greed_sentiment()
        else:
            self.cached_sentiment_data = self._get_kalshi_sentiment()
        
        return self.cached_sentiment_data
    
    def _get_fear_greed_sentiment(self) -> Dict[str, Any]:
        """从Fear & Greed Index获取情绪"""
        try:
            # 获取当前指数
            index_data = self.fear_greed_fetcher.get_current_index()
            value = index_data['value']
            classification = index_data['classification']
            
            # 获取7天和3天的趋势
            trend_7d = self.fear_greed_fetcher.get_trend(days=7)
            trend_3d = self.fear_greed_fetcher.get_trend(days=3)
            
            # 反向指标模式（默认）：恐慌时买入，贪婪时卖出
            if self.fear_greed_mode == "contrarian":
                # Fear & Greed: 0-100, 情绪得分: 0-1 (反向)
                # 极度恐慌(0) -> 1.0 (强烈看涨)
                # 极度贪婪(100) -> 0.0 (强烈看跌)
                base_sentiment = (100 - value) / 100.0
            else:
                # 动量模式：跟随趋势
                base_sentiment = value / 100.0
            
            # 趋势调整
            trend_boost = 0.0
            if self.fear_greed_mode == "contrarian":
                # 反向模式：恐慌中恢复 -> 更强买入信号
                trend_change = trend_7d.get('change', 0)
                if value < 50 and trend_change > 3:  # 恐慌中恢复
                    trend_boost = 0.15
                elif value > 50 and trend_change < -3:  # 贪婪中回落
                    trend_boost = -0.15
            
            sentiment_score = np.clip(base_sentiment + trend_boost, 0.0, 1.0)
            
            return {
                "sentiment_score": sentiment_score,
                "confidence": 0.85,  # Fear & Greed Index聚合多个数据源，可信度高
                "source": "fear_greed",
                "raw_value": int(value),  # 转换为Python int
                "classification": classification,
                "trend_7d": {k: int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in trend_7d.items()},
                "trend_3d": {k: int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in trend_3d.items()},
                "mode": self.fear_greed_mode
            }
            
        except Exception as e:
            logger.error(f"获取Fear & Greed情绪失败: {e}")
            return {
                "sentiment_score": 0.5,
                "confidence": 0.0,
                "source": "fear_greed",
                "error": str(e)
            }
    
    def _get_kalshi_sentiment(self) -> Dict[str, Any]:
        """从Kalshi获取情绪（备用）"""
        # 这里可以实现Kalshi的情绪计算逻辑
        logger.warning("Kalshi情绪计算尚未实现")
        return {
            "sentiment_score": 0.5,
            "confidence": 0.0,
            "source": "kalshi"
        }
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算技术指标"""
        if df is None or df.empty or len(df) < 20:
            logger.warning("数据不足，无法计算技术指标")
            return {}
        
        try:
            # 确保close列是数值类型
            df = df.copy()
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 移动平均线
            ma_short = df['close'].rolling(window=7).mean()
            ma_long = df['close'].rolling(window=25).mean()
            
            return {
                "rsi": rsi.iloc[-1],
                "ma_short": ma_short.iloc[-1],
                "ma_long": ma_long.iloc[-1],
                "current_price": df['close'].iloc[-1],
                "ma_trend": "bullish" if ma_short.iloc[-1] > ma_long.iloc[-1] else "bearish"
            }
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return {}
    
    def _apply_technical_filter(self, position: float, indicators: Dict) -> float:
        """应用技术指标过滤"""
        if not indicators:
            return position
        
        # RSI过滤
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if position > 0 and rsi > 70:  # 超买
                position *= 0.5
                logger.info(f"RSI超买({rsi:.2f})，降低多头仓位")
            elif position < 0 and rsi < 30:  # 超卖
                position *= 0.5
                logger.info(f"RSI超卖({rsi:.2f})，降低空头仓位")
        
        # 均线趋势过滤
        if 'ma_trend' in indicators:
            if position > 0 and indicators['ma_trend'] == 'bearish':
                position *= 0.7
                logger.info("均线空头排列，降低多头仓位")
            elif position < 0 and indicators['ma_trend'] == 'bullish':
                position *= 0.7
                logger.info("均线多头排列，降低空头仓位")
        
        return position
    
    def _apply_position_stability(self, new_position: float) -> float:
        """应用仓位稳定性机制"""
        now = datetime.now()
        
        # 如果是首次计算，直接返回
        if self.last_position_time is None:
            self.last_position = new_position
            self.last_position_time = now
            self.position_history.append({'time': now, 'position': new_position})
            return new_position
        
        # 计算距离上次调仓的时间
        time_since_last = (now - self.last_position_time).total_seconds() / 60  # 分钟
        
        # 计算仓位变化
        position_change = abs(new_position - self.last_position)
        
        # 判断是否需要调仓
        should_change = False
        reason = ""
        
        # 1. 如果变化很小，保持原仓位
        if position_change < self.position_change_threshold:
            logger.info(f"仓位变化过小({position_change:.3f})，保持原仓位 {self.last_position:.3f}")
            return self.last_position
        
        # 2. 如果持仓时间不足，需要更强的信号才能调仓
        if time_since_last < self.min_hold_minutes:
            if position_change >= self.position_change_threshold * 2:  # 需要2倍的变化才能提前调仓
                should_change = True
                reason = f"强烈信号(变化={position_change:.3f})，提前调仓"
            else:
                logger.info(f"持仓时间不足({time_since_last:.1f}分钟)，保持原仓位")
                return self.last_position
        else:
            should_change = True
            reason = f"正常调仓(变化={position_change:.3f})，距上次 {time_since_last:.1f}分钟"
        
        # 3. 直接使用新仓位（无平滑）
        if should_change:
            logger.info(f"调仓: {self.last_position:.3f} → {new_position:.3f}")
            logger.info(f"原因: {reason}")
            
            self.last_position = new_position
            self.last_position_time = now
            self.position_history.append({'time': now, 'position': new_position})
            
            return new_position
        
        return self.last_position
    
    def prepare_data(self, df, df_display, ma_dict, ma_dict_display, stats, args) -> Tuple[Optional[str], Optional[bytes]]:
        """准备数据（BasePredictor抽象方法）"""
        # 保存df供后续使用
        self.cached_df = df
        # 清空之前的sentiment缓存
        self.cached_sentiment_data = None
        # Kalshi策略不需要生成图表，直接返回空
        return "", None
    
    def generate_output(self, result: Dict[str, Any], stats: Dict[str, Any], args) -> Dict[str, Any]:
        """生成输出（BasePredictor抽象方法）"""
        prediction = result.get('position', 0.0)
        sentiment_data = self._get_market_sentiment()
        technical = result.get('technical_indicators', {})
        
        # ============ 详细指标展示 ============
        reasoning_parts = [
            "=" * 60,
            "📊 Kalshi策略详细指标分析",
            "=" * 60,
            "",
            "【1. Fear & Greed Index 组成（官方权重）】",
            "  - Volatility (波动率): 25%",
            "  - Market Momentum (动量): 25%",
            "  - Social Media (社交媒体): 15%",
            "  - Surveys (调查): 15%",
            "  - Market Dominance (主导地位): 10%",
            "  - Trends (搜索趋势): 10%",
            f"  综合指数: {sentiment_data.get('raw_value', 'N/A')}/100 ({sentiment_data.get('classification', 'N/A')})",
            ""
        ]
        
        # 趋势分析
        if 'trend_7d' in sentiment_data:
            trend_7d = sentiment_data['trend_7d']
            trend_3d = sentiment_data.get('trend_3d', {})
            reasoning_parts.extend([
                "【2. 趋势分析】",
                f"  - 7天变化: {trend_7d.get('change', 0):+d} ({trend_7d.get('direction', 'unknown')})",
                f"  - 3天变化: {trend_3d.get('change', 0):+d} ({trend_3d.get('direction', 'unknown')})",
                f"  - 趋势评估: {'恐慌中恢复' if sentiment_data.get('raw_value', 50) < 50 and trend_7d.get('change', 0) > 0 else '贪婪中回落' if sentiment_data.get('raw_value', 50) > 50 and trend_7d.get('change', 0) < 0 else '维持当前'}",
                ""
            ])
        
        # 策略计算过程
        raw_value = sentiment_data.get('raw_value', 50)
        base_sentiment = (100 - raw_value) / 100.0 if self.fear_greed_mode == 'contrarian' else raw_value / 100.0
        trend_boost = 0.0
        if self.fear_greed_mode == 'contrarian':
            trend_change = sentiment_data.get('trend_7d', {}).get('change', 0)
            if raw_value < 50 and trend_change > 3:
                trend_boost = 0.15
            elif raw_value > 50 and trend_change < -3:
                trend_boost = -0.15
        
        final_sentiment = np.clip(base_sentiment + trend_boost, 0.0, 1.0)
        
        reasoning_parts.extend([
            "【3. 情绪得分计算（我们的权重）】",
            f"  - 策略模式: {'反向指标 (Contrarian)' if self.fear_greed_mode == 'contrarian' else '动量跟随 (Momentum)'}",
            f"  - 基础情绪: {base_sentiment:.3f} (Fear&Greed反向映射)",
            f"  - 趋势加成: {trend_boost:+.3f} (7天变化>{3 if raw_value<50 else -3}: {'是' if abs(trend_boost) > 0 else '否'})",
            f"  - 最终情绪: {final_sentiment:.3f}",
            f"  - 置信度: {sentiment_data['confidence']:.3f} (固定)",
            ""
        ])
        
        # 仓位计算
        raw_position = 0.0
        position_calc_steps = []
        
        if final_sentiment >= self.sentiment_threshold_long:
            raw_position = (final_sentiment - 0.5) * 2
            raw_position = min(raw_position, self.max_position)
            position_calc_steps.append(f"  - 情绪 {final_sentiment:.3f} >= 阈值 {self.sentiment_threshold_long}")
            position_calc_steps.append(f"  - 基础仓位: ({final_sentiment:.3f} - 0.5) × 2 = {raw_position:.3f}")
        elif final_sentiment <= self.sentiment_threshold_short:
            raw_position = (final_sentiment - 0.5) * 2
            raw_position = max(raw_position, -self.max_position)
            position_calc_steps.append(f"  - 情绪 {final_sentiment:.3f} <= 阈值 {self.sentiment_threshold_short}")
            position_calc_steps.append(f"  - 基础仓位: ({final_sentiment:.3f} - 0.5) × 2 = {raw_position:.3f}")
        else:
            position_calc_steps.append(f"  - 情绪中性 ({self.sentiment_threshold_short} < {final_sentiment:.3f} < {self.sentiment_threshold_long})")
            position_calc_steps.append(f"  - 基础仓位: 0.000")
        
        # 技术指标调整
        tech_adjustment = 1.0
        if technical:
            if 'rsi' in technical:
                rsi = technical['rsi']
                if raw_position > 0 and rsi > 70:
                    tech_adjustment *= 0.5
                    position_calc_steps.append(f"  - RSI超买调整: {rsi:.2f} > 70, 仓位×0.5")
                elif raw_position < 0 and rsi < 30:
                    tech_adjustment *= 0.5
                    position_calc_steps.append(f"  - RSI超卖调整: {rsi:.2f} < 30, 仓位×0.5")
            
            if 'ma_trend' in technical:
                if raw_position > 0 and technical['ma_trend'] == 'bearish':
                    tech_adjustment *= 0.7
                    position_calc_steps.append(f"  - 均线空头调整: 仓位×0.7")
                elif raw_position < 0 and technical['ma_trend'] == 'bullish':
                    tech_adjustment *= 0.7
                    position_calc_steps.append(f"  - 均线多头调整: 仓位×0.7")
        
        # 置信度调整
        position_calc_steps.append(f"  - 置信度调整: ×{sentiment_data['confidence']:.3f}")
        position_calc_steps.append(f"  - 技术调整: ×{tech_adjustment:.3f}")
        adjusted_position = raw_position * sentiment_data['confidence'] * tech_adjustment
        position_calc_steps.append(f"  - 调整后仓位: {adjusted_position:.3f}")
        
        reasoning_parts.extend([
            "【4. 仓位计算过程】",
            *position_calc_steps,
            ""
        ])
        
        # 技术指标详情
        if technical:
            reasoning_parts.extend([
                "【5. 技术指标当前值】",
                f"  - RSI(14): {technical.get('rsi', 'N/A'):.2f}",
                f"  - MA(7): {technical.get('ma_short', 'N/A'):.2f}",
                f"  - MA(25): {technical.get('ma_long', 'N/A'):.2f}",
                f"  - 当前价格: ${technical.get('current_price', 'N/A'):.2f}",
                f"  - 均线趋势: {technical.get('ma_trend', 'N/A')}",
                ""
            ])
        else:
            reasoning_parts.extend([
                "【5. 技术指标当前值】",
                "  ⚠️ K线数据不足，无法计算技术指标",
                ""
            ])
        
        # 最终决策
        reasoning_parts.extend([
            "【6. 最终决策】",
            f"  - 建议仓位: {prediction:.3f}",
            f"  - 操作建议: {self._get_action_description(prediction)}",
        ])
        
        # 决策理由
        if prediction > 0.3:
            if self.fear_greed_mode == "contrarian":
                reasoning_parts.append(f"  - 决策理由: 市场恐慌({raw_value}/100)，反向买入机会")
            else:
                reasoning_parts.append(f"  - 决策理由: 市场乐观({raw_value}/100)，跟随做多")
        elif prediction < -0.3:
            if self.fear_greed_mode == "contrarian":
                reasoning_parts.append(f"  - 决策理由: 市场贪婪({raw_value}/100)，反向卖出")
            else:
                reasoning_parts.append(f"  - 决策理由: 市场恐慌({raw_value}/100)，跟随做空")
        else:
            reasoning_parts.append(f"  - 决策理由: 信号不明确，建议观望")
        
        reasoning_parts.append("=" * 60)
        
        reasoning = "\n".join(reasoning_parts)
        
        # 将详细分析写入日志，方便复盘
        logger.info("=" * 60)
        logger.info("📊 Kalshi策略详细分析报告")
        logger.info("=" * 60)
        for line in reasoning_parts:
            if line:  # 跳过空行
                logger.info(line)
        logger.info("=" * 60)
        
        return {
            "symbol": self.symbol,
            "strategy": "kalshi",
            "timestamp": datetime.now().isoformat(),
            "position": float(prediction),
            "confidence": float(sentiment_data['confidence']),
            "reasoning": reasoning,
            "action": self._get_action_description(prediction),
            "chart_data": None,
            # 添加交易所需的额外信息
            "sentiment": sentiment_data,
            "technical_indicators": result.get('technical_indicators', {}),
            # 从technical_indicators中提取current_price到顶层
            "current_price": result.get('technical_indicators', {}).get('current_price', 0.0)
        }


def main():
    """测试函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # 创建交易器
    trader = KalshiTrader(symbol="BTCUSDC", name="test")
    
    # 分析市场
    logger.info("=== 开始分析 ===")
    result = trader.analyze(interval="1h", days=7)
    
    # 输出结果
    logger.info("\n=== 分析结果 ===")
    logger.info(f"交易对: {result['symbol']}")
    logger.info(f"建议仓位: {result['position']:.3f}")
    logger.info(f"操作建议: {result['action']}")
    logger.info(f"情绪得分: {result['sentiment']['sentiment_score']:.3f}")
    logger.info(f"置信度: {result['sentiment']['confidence']:.3f}")
    
    if 'technical_indicators' in result:
        ti = result['technical_indicators']
        if 'rsi' in ti:
            logger.info(f"RSI: {ti['rsi']:.2f}")
        if 'current_price' in ti:
            logger.info(f"当前价格: ${ti['current_price']:.2f}")


if __name__ == "__main__":
    main()
