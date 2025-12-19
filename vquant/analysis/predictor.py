#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Quantitative Predictor - 基于 funding rate 和技术指标的预测模型
预测涨跌并输出仓位建议
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)


class QuantPredictor:
    """
    量化预测器 - 基于多个因子综合评分预测市场方向

    评分因子：
    1. Funding Rate - 资金费率反映多空力量对比
    2. Momentum - 价格动量趋势
    3. Impulse - 动量加速度（冲量）
    4. MA趋势 - 价格相对MA7位置
    5. RSI - 超买超卖指标
    6. MACD - MACD金叉死叉
    7. Volume - 成交量确认
    """

    def __init__(self, symbol: str = "BTCUSDC", config_dir: str = "config"):
        """
        初始化预测器
        
        Args:
            symbol: 交易对符号（用于加载对应的阈值配置）
            config_dir: 配置文件目录
        """
        self.symbol = symbol
        self.weights = {
            "funding_rate": 0.20,  # 资金费率权重
            "momentum": 0.20,      # 动量权重
            "impulse": 0.15,       # 冲量权重（动量加速度）
            "ma_trend": 0.15,      # 均线趋势权重
            "rsi": 0.12,           # RSI权重
            "macd": 0.10,          # MACD权重
            "volume": 0.08,        # 成交量权重
        }
        
        # 加载阈值配置（必须存在，否则抛出异常）
        self.thresholds = self._load_thresholds(symbol, config_dir)
    
    def _load_thresholds(self, symbol: str, config_dir: str) -> Dict:
        """
        从JSON文件加载阈值配置
        
        Args:
            symbol: 交易对符号
            config_dir: 配置文件目录
            
        Returns:
            Threshold configuration dictionary
            
        Raises:
            FileNotFoundError: Config file does not exist
            ValueError: Config file format error
        """
        config_path = Path(config_dir) / f"thresholds_{symbol.lower()}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Threshold config file not found: {config_path}\n"
                f"Please run calibrator first: python -m vquant.model.calibrator --symbol {symbol}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'thresholds' not in data:
                raise ValueError(f"Config file format error: missing 'thresholds' field")
            
            logger.info(f"Loaded threshold config: {config_path} (updated: {data.get('updated_at', 'unknown')})")
            return data['thresholds']
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file JSON format error: {e}")
    def predict(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于市场数据预测涨跌并给出仓位建议

        Args:
            stats: 包含技术指标和市场数据的字典
                - funding_rate: 当前资金费率
                - momentum: 价格动量（近期均价相对早期均价的变化率）
                - current_ma7: 当前MA7值
                - current_price: 当前价格
                - current_rsi: 当前RSI值
                - current_macd: 当前MACD值
                - current_signal: 当前MACD信号线值
                - volume_strength: 成交量强度

        Returns:
            {
                "position": float,  # -1.0 到 1.0 的仓位建议
                "confidence": str,  # "low", "medium", "high"
                "score": float,     # 综合评分 -100 到 100
                "factors": dict,    # 各因子得分详情
                "reasoning": str,   # 决策理由
            }
        """
        logger.info("开始预测市场方向...")

        # 计算各因子得分
        factors = {}

        # 1. Funding Rate 评分 (-100 到 100)
        factors["funding_rate"] = self._score_funding_rate(stats.get("funding_rate", 0))

        # 2. Momentum 评分 (-100 到 100)
        factors["momentum"] = self._score_momentum(
            stats.get("momentum", 0)
        )

        # 3. Impulse 评分 (-100 到 100)
        factors["impulse"] = self._score_impulse(
            stats.get("impulse", 0)
        )

        # 4. MA 趋势评分 (-100 到 100)
        factors["ma_trend"] = self._score_ma_trend(
            stats.get("current_ma7"), stats.get("current_price")
        )

        # 5. RSI 评分 (-100 到 100)
        factors["rsi"] = self._score_rsi(stats.get("current_rsi"))

        # 6. MACD 评分 (-100 到 100)
        factors["macd"] = self._score_macd(
            stats.get("current_macd"), stats.get("current_signal")
        )

        # 7. Volume 评分 (-100 到 100)
        factors["volume"] = self._score_volume(stats.get("volume_strength", 0))

        # 计算加权综合得分
        total_score = sum(factors[key] * self.weights[key] for key in factors.keys())

        # 将得分转换为仓位建议 (-1.0 到 1.0)
        position = self._score_to_position(total_score)

        # 评估置信度
        confidence = self._calculate_confidence(factors, total_score)

        # 生成决策理由
        reasoning = self._generate_reasoning(factors, total_score, stats)

        result = {
            "position": round(position, 2),
            "confidence": confidence,
            "score": round(total_score, 2),
            "factors": {k: round(v, 2) for k, v in factors.items()},
            "reasoning": reasoning,
        }

        logger.info(
            f"预测完成: position={result['position']}, confidence={result['confidence']}"
        )
        logger.debug(f"因子得分: {result['factors']}")

        return result

    def _score_funding_rate(self, funding_rate: Optional[float]) -> float:
        """
        资金费率评分（基于DOGE/BTC实际历史数据调整）
        
        实际数据分析（100期历史）：
        - 平均值：0.003%
        - 90分位：0.009%
        - 10分位：-0.002%
        - 最大值：0.010%
        - 最小值：-0.011%
        - 100%数据在 -0.02% 到 0.02% 之间
        
        资金费率正值：多头支付空头，市场偏多 -> 可能过热
        资金费率负值：空头支付多头，市场偏空 -> 可能超跌
        """
        if funding_rate is None:
            return 0.0
        
        # 基于实际数据分布设置阈值
        # 90分位约0.009%，10分位约-0.002%
        if funding_rate > 0.010:
            return -100.0  # 极度罕见，强烈看跌
        elif funding_rate > 0.008:
            return -80.0   # >90分位，多头过热
        elif funding_rate > 0.006:
            return -50.0   # >75分位，多头偏强
        elif funding_rate > 0.003:
            return -20.0   # >中位数，多头略强
        elif funding_rate > -0.002:
            return 0.0     # 中性区间（10分位到中位数之间）
        elif funding_rate > -0.005:
            return 30.0    # <10分位，空头偏强
        elif funding_rate > -0.008:
            return 60.0    # 空头较强
        elif funding_rate > -0.010:
            return 80.0    # 接近历史最小值
        else:
            return 100.0   # 极度罕见，强烈看涨
    
    def _score_momentum(self, momentum: float) -> float:
        """
        动量评分（基于配置文件中的阈值）
        
        momentum: 近期均价相对早期均价的变化率（%）
        """
        t = self.thresholds.get('momentum', {})
        
        if momentum > t.get('extreme_bullish', 4):
            return 100.0   # 极强上涨
        elif momentum > t.get('strong_bullish', 2.5):
            return 80.0    # 强势上涨
        elif momentum > t.get('bullish', 1.0):
            return 50.0    # 上涨
        elif momentum > t.get('neutral_high', 0):
            return 20.0    # 微涨
        elif momentum > t.get('neutral_low', -1.5):
            return -20.0   # 微跌
        elif momentum > t.get('bearish', -3.0):
            return -50.0   # 下跌
        elif momentum > t.get('strong_bearish', -4.0):
            return -80.0   # 强势下跌
        else:
            return -100.0  # 极弱下跌

    def _score_impulse(self, impulse: float) -> float:
        """
        冲量评分（基于配置文件中的阈值）
        
        Impulse measures momentum acceleration - indicates if momentum is speeding up or slowing down
        Positive impulse: momentum accelerating (bullish)
        Negative impulse: momentum decelerating (bearish)
        """
        t = self.thresholds.get('impulse', {})
        
        if impulse > t.get('extreme_bullish', 5):
            return 100.0   # 极强加速
        elif impulse > t.get('strong_bullish', 3):
            return 80.0    # 强势加速
        elif impulse > t.get('bullish', 1.5):
            return 50.0    # 加速
        elif impulse > t.get('neutral_high', 0):
            return 20.0    # 微加速
        elif impulse > t.get('neutral_low', -1.5):
            return -20.0   # 微减速
        elif impulse > t.get('bearish', -3):
            return -50.0   # 减速
        elif impulse > t.get('strong_bearish', -5):
            return -80.0   # 强势减速
        else:
            return -100.0  # 极弱减速

    def _score_ma_trend(
        self, ma7: Optional[float], current_price: Optional[float]
    ) -> float:
        """
        MA7趋势评分（基于配置文件中的阈值）

        价格在MA7上方：趋势向上
        价格在MA7下方：趋势向下
        """
        if ma7 is None or current_price is None:
            return 0.0

        # 计算价格偏离MA7的程度
        deviation = (current_price - ma7) / ma7 * 100 if ma7 > 0 else 0

        # 使用配置的阈值评分
        t = self.thresholds.get('ma_deviation', {})
        
        if deviation > t.get('extreme_bullish', 1.5):
            return 100.0   # 极强上涨
        elif deviation > t.get('strong_bullish', 1.0):
            return 80.0    # 强势上涨
        elif deviation > t.get('bullish', 0.4):
            return 50.0    # 上涨
        elif deviation > t.get('neutral_high', 0):
            return 20.0    # 微涨
        elif deviation > t.get('neutral_low', -0.4):
            return -20.0   # 微跌
        elif deviation > t.get('bearish', -1.0):
            return -50.0   # 下跌
        elif deviation > t.get('strong_bearish', -2.0):
            return -80.0   # 强势下跌
        else:
            return -100.0  # 极弱下跌

    def _score_rsi(self, rsi: Optional[float]) -> float:
        """
        RSI 评分（基于配置文件中的阈值）

        RSI高：超买，可能回调
        RSI低：超卖，可能反弹
        """
        if rsi is None:
            return 0.0

        t = self.thresholds.get('rsi', {})
        
        if rsi > t.get('extreme_overbought', 79):
            return -100.0  # 极度超买
        elif rsi > t.get('overbought', 73):
            return -80.0   # 超买
        elif rsi > t.get('slightly_overbought', 61):
            return -50.0   # 偏强
        elif rsi > t.get('neutral_high', 48):
            return -20.0   # 略强
        elif rsi > t.get('neutral_low', 35):
            return 20.0    # 略弱
        elif rsi > t.get('slightly_oversold', 20):
            return 50.0    # 偏弱
        elif rsi > t.get('oversold', 16):
            return 80.0    # 超卖
        else:
            return 100.0   # 极度超卖

    def _score_macd(self, macd: Optional[float], signal: Optional[float]) -> float:
        """
        MACD 评分（基于配置文件中的阈值）

        MACD > Signal: 金叉，看涨
        MACD < Signal: 死叉，看跌
        """
        if macd is None or signal is None:
            return 0.0

        diff = macd - signal
        t = self.thresholds.get('macd_diff', {})

        if diff > t.get('extreme_bullish', 0.004):
            return 100.0        # 极强金叉
        elif diff > t.get('strong_bullish', 0.003):
            return 80.0         # 强金叉
        elif diff > t.get('bullish', 0.0015):
            return 50.0         # 金叉
        elif diff > t.get('neutral_high', 0):
            return 20.0         # 弱金叉
        elif diff > t.get('neutral_low', -0.0014):
            return -20.0        # 弱死叉
        elif diff > t.get('bearish', -0.003):
            return -50.0        # 死叉
        elif diff > t.get('strong_bearish', -0.004):
            return -80.0        # 强死叉
        else:
            return -100.0       # 极强死叉

    def _score_volume(self, volume_strength: float) -> float:
        """
        成交量评分（基于配置文件中的阈值）

        volume_strength: 相对20周期均值的变化百分比
        放量确认趋势，缩量趋势减弱
        """
        t = self.thresholds.get('volume_change', {})
        
        if volume_strength > t.get('extreme_volume', 230):
            return 100.0   # 极度放量
        elif volume_strength > t.get('high_volume', 120):
            return 80.0    # 显著放量
        elif volume_strength > t.get('above_average', 20):
            return 50.0    # 放量
        elif volume_strength > t.get('neutral', -30):
            return 0.0     # 中性区间
        elif volume_strength > t.get('below_average', -55):
            return -30.0   # 缩量
        elif volume_strength > t.get('low_volume', -70):
            return -60.0   # 显著缩量
        else:
            return -80.0   # 极度缩量

    def _score_to_position(self, score: float) -> float:
        """
        将评分转换为仓位建议

        score: -100 到 100
        position: -1.0 到 1.0

        使用非线性映射，放大极端信号
        """
        # 线性映射
        position = score / 100.0

        # 非线性调整：放大强信号，抑制弱信号
        if abs(position) > 0.6:
            # 强信号：保持或放大
            position = position * 1.1
        elif abs(position) < 0.2:
            # 弱信号：进一步抑制
            position = position * 0.5

        # 限制在 -1.0 到 1.0
        return max(-1.0, min(1.0, position))

    def _calculate_confidence(
        self, factors: Dict[str, float], total_score: float
    ) -> str:
        """
        计算置信度

        high: 因子方向一致性高，且总分绝对值大
        medium: 因子方向部分一致，或总分中等
        low: 因子方向分歧大，或总分接近0
        """
        # 统计正负因子
        positive = sum(1 for v in factors.values() if v > 20)
        negative = sum(1 for v in factors.values() if v < -20)
        neutral = len(factors) - positive - negative

        # 方向一致性
        consistency = max(positive, negative) / len(factors)

        # 综合评估
        abs_score = abs(total_score)

        if consistency > 0.75 and abs_score > 50:
            return "high"
        elif consistency > 0.5 and abs_score > 30:
            return "medium"
        else:
            return "low"

    def _generate_reasoning(
        self, factors: Dict[str, float], total_score: float, stats: Dict[str, Any]
    ) -> str:
        """生成决策理由 - 结构化输出：结论 + 支持因子 + 对冲因子"""

        # 分类因子：看多、看空、中性
        bullish_factors = []
        bearish_factors = []
        neutral_factors = []

        # Funding Rate（基于实际数据分布调整）
        fr = stats.get("funding_rate", 0)
        if fr is not None:
            if fr > 0.008:
                bearish_factors.append(f"资金费率{fr:.4f}%过高（>90分位），多头过热")
            elif fr > 0.006:
                bearish_factors.append(f"资金费率{fr:.4f}%偏高（>75分位）")
            elif fr > 0.003:
                bearish_factors.append(f"资金费率{fr:.4f}%略高")
            elif fr < -0.005:
                bullish_factors.append(f"资金费率{fr:.4f}%偏低，空头占优")
            elif fr < -0.002:
                bullish_factors.append(f"资金费率{fr:.4f}%略低（<10分位）")
            else:
                neutral_factors.append(f"资金费率{fr:.4f}%中性")

        # Momentum
        momentum = stats.get("momentum", 0)
        if momentum > 2:
            bullish_factors.append(f"动量强劲+{momentum:.1f}%")
        elif momentum > 0.5:
            bullish_factors.append(f"动量向上+{momentum:.1f}%")
        elif momentum < -2:
            bearish_factors.append(f"动量下跌{momentum:.1f}%")
        elif momentum < -0.5:
            bearish_factors.append(f"动量向下{momentum:.1f}%")

        # MA7 趋势
        ma7 = stats.get("current_ma7")
        current_price = stats.get("current_price")
        if ma7 and current_price:
            deviation = (current_price - ma7) / ma7 * 100 if ma7 > 0 else 0
            if deviation > 2:
                bullish_factors.append(f"价格高于MA7 +{deviation:.1f}%")
            elif deviation > 0.5:
                bullish_factors.append(f"价格略高于MA7 +{deviation:.1f}%")
            elif deviation < -2:
                bearish_factors.append(f"价格低于MA7 {deviation:.1f}%")
            elif deviation < -0.5:
                bearish_factors.append(f"价格略低于MA7 {deviation:.1f}%")
            else:
                neutral_factors.append("价格贴近MA7")

        # RSI
        rsi = stats.get("current_rsi")
        if rsi:
            if rsi > 70:
                bearish_factors.append(f"RSI{rsi:.1f}超买")
            elif rsi > 60:
                bearish_factors.append(f"RSI{rsi:.1f}偏高")
            elif rsi < 30:
                bullish_factors.append(f"RSI{rsi:.1f}超卖")
            elif rsi < 40:
                bullish_factors.append(f"RSI{rsi:.1f}偏低")

        # MACD
        macd = stats.get("current_macd")
        signal = stats.get("current_signal")
        if macd and signal:
            if macd > signal:
                if macd > 0:
                    bullish_factors.append("MACD强势金叉")
                else:
                    bullish_factors.append("MACD弱势金叉")
            else:
                if macd > 0:
                    bearish_factors.append("MACD弱势死叉")
                else:
                    bearish_factors.append("MACD强势死叉")

        # Volume
        vol_strength = stats.get("volume_strength", 0)
        if vol_strength > 50:
            bullish_factors.append(f"成交量放大{vol_strength:.0f}%")
        elif vol_strength < -50:
            bearish_factors.append(f"成交量萎缩{vol_strength:.0f}%")

        # 构建reasoning
        if total_score > 50:
            conclusion = "【做多】综合评分强势看多"
        elif total_score > 20:
            conclusion = "【小仓做多】综合评分偏多"
        elif total_score < -50:
            conclusion = "【做空】综合评分强势看空"
        elif total_score < -20:
            conclusion = "【小仓做空】综合评分偏空"
        else:
            conclusion = "【观望】综合评分中性"

        parts = [conclusion]

        if bullish_factors:
            parts.append("看多信号：" + "、".join(bullish_factors))

        if bearish_factors:
            parts.append("看空信号：" + "、".join(bearish_factors))

        if neutral_factors:
            parts.append("中性：" + "、".join(neutral_factors))

        # 如果是中性观望，特别说明为什么
        if abs(total_score) <= 20 and (bullish_factors or bearish_factors):
            parts.append("多空信号相互抵消，暂不具备明确方向")

        return "。".join(parts) + "。"


if __name__ == "__main__":
    # 测试用例
    logging.basicConfig(level=logging.INFO)

    predictor = QuantPredictor()

    # 模拟数据
    test_stats = {
        "funding_rate": 0.08,  # 资金费率偏高
        "momentum": -3.5,           # 动量下跌
        "current_ma7": 100.5,
        "current_price": 98.0,  # 价格低于MA7
        "current_rsi": 65.0,
        "current_macd": 0.5,
        "current_signal": 0.3,
        "volume_strength": 80.0,
    }

    result = predictor.predict(test_stats)

    print("\n=== 预测结果 ===")
    print(f"Position: {result['position']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Score: {result['score']}")
    print(f"Factors: {result['factors']}")
    print(f"Reasoning: {result['reasoning']}")
