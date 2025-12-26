#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prefetch 1-minute historical data for model training and validation
Data will be aggregated to 1h, 4h, or 1d during training
"""

import logging
from vquant.data import prefetch_all_data


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 配置要拉取的交易对
SYMBOLS = [
    'BTCUSDC',
    'ETHUSDC',
    'BNBUSDC',
    'SOLUSDC',
    'DOGEUSDC',
]

# 起始日期（从2023年1月1日至今，约724天）
START_DATE = '2023-01-01'

# 请求延迟（秒），避免触发限流
REQUEST_DELAY = 0.5  # 每个批次延迟0.5秒


if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("开始预拉取1分钟历史数据")
        logger.info(f"交易对: {', '.join(SYMBOLS)}")
        logger.info(f"起始日期: {START_DATE}")
        logger.info(f"请求延迟: {REQUEST_DELAY}秒/批次")
        logger.info("="*60)
        logger.warning("注意: 1分钟数据量很大，请耐心等待...")
        logger.warning("预计每个交易对需要约10-30分钟，请勿中断...")
        logger.info("="*60)
        
        stats = prefetch_all_data(
            symbols=SYMBOLS,
            start_date=START_DATE,
            request_delay=REQUEST_DELAY
        )
        
        logger.info("\n✅ 全部完成！")
        logger.info(f"成功: {stats['success']}/{stats['total']}")
        logger.info("现在可以使用 resample_klines() 函数聚合为1h/4h/1d数据进行训练")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  用户中断")
    except Exception as e:
        logger.error(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
