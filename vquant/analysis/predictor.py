#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Quantitative Predictor - 基于 funding rate 和技术指标的预测模型
预测涨跌并输出仓位建议
"""

import logging
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)


class QuantPredictor:
    """
    量化预测器 - 基于多个因子综合评分预测市场方向
    
    评分因子：
    1. Funding Rate - 资金费率反映多空力量对比
    2. MA7 拐点 - 短期趋势转折信号
    3. MA趋势 - 均线多空排列
    4. RSI - 超买超卖指标
    5. MACD - 动量指标
    6. Volume - 成交量确认
    """
    
    def __init__(self):
        """初始化预测器"""
        self.weights = {
            'funding_rate': 0.25,      # 资金费率权重
            'ma7_inflection': 0.25,    # MA7拐点权重
            'ma_trend': 0.15,          # 均线趋势权重
            'rsi': 0.15,               # RSI权重
            'macd': 0.10,              # MACD权重
            'volume': 0.10,            # 成交量权重
        }
    
    def predict(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于市场数据预测涨跌并给出仓位建议
        
        Args:
            stats: 包含技术指标和市场数据的字典
                - funding_rate: 当前资金费率
                - ma7_inflection: MA7拐点状态 ('upward', 'downward', 'continuing')
                - current_ma7: 当前MA7值
                - current_ma25: 当前MA25值
                - current_ma99: 当前MA99值
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
        factors['funding_rate'] = self._score_funding_rate(
            stats.get('funding_rate', 0)
        )
        
        # 2. MA7 拐点评分 (-100 到 100)
        factors['ma7_inflection'] = self._score_ma7_inflection(
            stats.get('ma7_inflection', 'continuing')
        )
        
        # 3. MA 趋势评分 (-100 到 100)
        factors['ma_trend'] = self._score_ma_trend(
            stats.get('current_ma7'),
            stats.get('current_ma25'),
            stats.get('current_ma99')
        )
        
        # 4. RSI 评分 (-100 到 100)
        factors['rsi'] = self._score_rsi(
            stats.get('current_rsi')
        )
        
        # 5. MACD 评分 (-100 到 100)
        factors['macd'] = self._score_macd(
            stats.get('current_macd'),
            stats.get('current_signal')
        )
        
        # 6. Volume 评分 (-100 到 100)
        factors['volume'] = self._score_volume(
            stats.get('volume_strength', 0)
        )
        
        # 计算加权综合得分
        total_score = sum(
            factors[key] * self.weights[key] 
            for key in factors.keys()
        )
        
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
        
        logger.info(f"预测完成: position={result['position']}, confidence={result['confidence']}")
        logger.debug(f"因子得分: {result['factors']}")
        
        return result
    
    def _score_funding_rate(self, funding_rate: Optional[float]) -> float:
        """
        资金费率评分
        
        资金费率正值：多头支付空头，市场看多 -> 负分（可能过热）
        资金费率负值：空头支付多头，市场看空 -> 正分（可能超跌）
        
        极端值：
        > 0.1%: 多头过热，看跌 -> -100
        > 0.05%: 多头偏多，看跌 -> -50
        -0.05% to 0.05%: 中性 -> 0
        < -0.05%: 空头偏多，看涨 -> +50
        < -0.1%: 空头过热，看涨 -> +100
        """
        if funding_rate is None:
            return 0.0
        
        # 反向逻辑：资金费率越高（多头支付多），越可能回调
        if funding_rate > 0.15:
            return -100.0
        elif funding_rate > 0.10:
            return -80.0
        elif funding_rate > 0.05:
            return -50.0
        elif funding_rate > 0.02:
            return -20.0
        elif funding_rate > -0.02:
            return 0.0
        elif funding_rate > -0.05:
            return 20.0
        elif funding_rate > -0.10:
            return 50.0
        elif funding_rate > -0.15:
            return 80.0
        else:
            return 100.0
    
    def _score_ma7_inflection(self, inflection: str) -> float:
        """
        MA7 拐点评分
        
        upward: 短期趋势转多 -> +80
        downward: 短期趋势转空 -> -80
        continuing: 延续趋势 -> 0
        """
        if inflection == 'upward':
            return 80.0
        elif inflection == 'downward':
            return -80.0
        else:
            return 0.0
    
    def _score_ma_trend(
        self, 
        ma7: Optional[float], 
        ma25: Optional[float], 
        ma99: Optional[float]
    ) -> float:
        """
        均线趋势评分
        
        多头排列 (MA7 > MA25 > MA99): +100
        空头排列 (MA7 < MA25 < MA99): -100
        混乱排列: 0
        """
        if ma7 is None or ma25 is None or ma99 is None:
            return 0.0
        
        # 多头排列
        if ma7 > ma25 > ma99:
            return 100.0
        # 空头排列
        elif ma7 < ma25 < ma99:
            return -100.0
        # 金叉区域
        elif ma7 > ma25:
            return 50.0
        # 死叉区域
        elif ma7 < ma25:
            return -50.0
        else:
            return 0.0
    
    def _score_rsi(self, rsi: Optional[float]) -> float:
        """
        RSI 评分
        
        RSI > 70: 超买 -> -80
        RSI 50-70: 偏强 -> +30
        RSI 30-50: 偏弱 -> -30
        RSI < 30: 超卖 -> +80
        """
        if rsi is None:
            return 0.0
        
        if rsi > 80:
            return -100.0
        elif rsi > 70:
            return -80.0
        elif rsi > 60:
            return -40.0
        elif rsi > 50:
            return 0.0
        elif rsi > 40:
            return 0.0
        elif rsi > 30:
            return 40.0
        elif rsi > 20:
            return 80.0
        else:
            return 100.0
    
    def _score_macd(
        self, 
        macd: Optional[float], 
        signal: Optional[float]
    ) -> float:
        """
        MACD 评分
        
        MACD > Signal 且都为正: 强多头 -> +100
        MACD > Signal 且都为负: 弱多头 -> +50
        MACD < Signal 且都为正: 弱空头 -> -50
        MACD < Signal 且都为负: 强空头 -> -100
        """
        if macd is None or signal is None:
            return 0.0
        
        diff = macd - signal
        
        if macd > 0 and signal > 0:
            # 都为正，看差值
            if diff > 0:
                return min(100.0, diff * 1000)  # 金叉向上
            else:
                return max(-50.0, diff * 1000)  # 死叉向下
        elif macd < 0 and signal < 0:
            # 都为负，看差值
            if diff > 0:
                return min(50.0, diff * 1000)   # 金叉向上
            else:
                return max(-100.0, diff * 1000) # 死叉向下
        else:
            # 一正一负，过渡区
            return diff * 500
    
    def _score_volume(self, volume_strength: float) -> float:
        """
        成交量评分
        
        volume_strength > 50%: 放量 -> 趋势确认
        volume_strength < -50%: 缩量 -> 趋势减弱
        """
        if volume_strength > 100:
            return 80.0
        elif volume_strength > 50:
            return 50.0
        elif volume_strength > 0:
            return 20.0
        elif volume_strength > -50:
            return -20.0
        else:
            return -50.0
    
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
        self, 
        factors: Dict[str, float], 
        total_score: float
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
        self, 
        factors: Dict[str, float], 
        total_score: float,
        stats: Dict[str, Any]
    ) -> str:
        """生成决策理由"""
        reasons = []
        
        # Funding Rate
        fr = stats.get('funding_rate', 0)
        if fr is not None:
            if fr > 0.10:
                reasons.append(f"资金费率高达 {fr:.4f}%，多头过热，存在回调风险")
            elif fr > 0.05:
                reasons.append(f"资金费率 {fr:.4f}% 偏高，多头占优但需谨慎")
            elif fr < -0.10:
                reasons.append(f"资金费率 {fr:.4f}% 极度负值，空头过度，反弹概率大")
            elif fr < -0.05:
                reasons.append(f"资金费率 {fr:.4f}% 为负，空头占优，可能超跌反弹")
            else:
                reasons.append(f"资金费率 {fr:.4f}% 处于中性区间")
        
        # MA7 拐点
        inflection = stats.get('ma7_inflection', 'continuing')
        if inflection == 'upward':
            reasons.append("MA7 出现向上拐点，短期趋势转多")
        elif inflection == 'downward':
            reasons.append("MA7 出现向下拐点，短期趋势转空")
        
        # MA 趋势
        ma7 = stats.get('current_ma7')
        ma25 = stats.get('current_ma25')
        ma99 = stats.get('current_ma99')
        if ma7 and ma25 and ma99:
            if ma7 > ma25 > ma99:
                reasons.append("均线呈多头排列，趋势向上")
            elif ma7 < ma25 < ma99:
                reasons.append("均线呈空头排列，趋势向下")
            elif ma7 > ma25:
                reasons.append("MA7 上穿 MA25，短期看多")
            elif ma7 < ma25:
                reasons.append("MA7 下穿 MA25，短期看空")
        
        # RSI
        rsi = stats.get('current_rsi')
        if rsi:
            if rsi > 70:
                reasons.append(f"RSI {rsi:.1f} 超买，存在回调压力")
            elif rsi < 30:
                reasons.append(f"RSI {rsi:.1f} 超卖，存在反弹机会")
        
        # MACD
        macd = stats.get('current_macd')
        signal = stats.get('current_signal')
        if macd and signal:
            if macd > signal and macd > 0:
                reasons.append("MACD 金叉向上，动量强劲")
            elif macd < signal and macd < 0:
                reasons.append("MACD 死叉向下，动量减弱")
        
        # Volume
        vol_strength = stats.get('volume_strength', 0)
        if vol_strength > 50:
            reasons.append(f"成交量放大 {vol_strength:.0f}%，趋势得到确认")
        elif vol_strength < -50:
            reasons.append(f"成交量萎缩 {vol_strength:.0f}%，趋势可能转弱")
        
        # 总结
        if total_score > 50:
            summary = "综合评分偏多，建议做多"
        elif total_score > 20:
            summary = "综合评分偏多，建议小仓位做多"
        elif total_score < -50:
            summary = "综合评分偏空，建议做空"
        elif total_score < -20:
            summary = "综合评分偏空，建议小仓位做空"
        else:
            summary = "综合评分中性，建议观望"
        
        return summary + "。" + "；".join(reasons) + "。"


if __name__ == '__main__':
    # 测试用例
    logging.basicConfig(level=logging.INFO)
    
    predictor = QuantPredictor()
    
    # 模拟数据
    test_stats = {
        'funding_rate': 0.08,       # 资金费率偏高
        'ma7_inflection': 'downward',  # MA7 向下拐点
        'current_ma7': 100.5,
        'current_ma25': 102.0,
        'current_ma99': 105.0,
        'current_rsi': 65.0,
        'current_macd': 0.5,
        'current_signal': 0.3,
        'volume_strength': 80.0,
    }
    
    result = predictor.predict(test_stats)
    
    print("\n=== 预测结果 ===")
    print(f"Position: {result['position']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Score: {result['score']}")
    print(f"Factors: {result['factors']}")
    print(f"Reasoning: {result['reasoning']}")
