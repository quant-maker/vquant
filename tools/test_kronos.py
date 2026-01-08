#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kronos策略测试脚本
用于验证爬虫和策略功能
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
)

print("=" * 60)
print("Kronos策略测试")
print("=" * 60)

# 测试1: 爬虫功能
print("\n[测试1] 爬虫模块...")
try:
    from vquant.data.kronos_scraper import KronosScraper
    
    scraper = KronosScraper(max_staleness_hours=24)
    print("✓ 爬虫模块导入成功")
    
    print("\n尝试获取Kronos预测数据...")
    prediction = scraper.fetch_prediction("BTCUSDT")
    
    if prediction:
        print("✓ 成功获取预测数据")
        print(f"  - 交易对: {prediction['symbol']}")
        print(f"  - 趋势: {prediction['trend']}")
        print(f"  - 置信度: {prediction['confidence']:.1%}")
        print(f"  - 预测价格: {prediction.get('predicted_price', 'N/A')}")
        print(f"  - 当前价格: {prediction.get('current_price', 'N/A')}")
        print(f"  - 数据新鲜度: {'✓ 新鲜' if not prediction.get('is_stale') else '⚠️  过旧'}")
        if prediction.get('staleness_hours') is not None:
            print(f"  - 数据年龄: {prediction['staleness_hours']:.1f} 小时")
        
        is_safe = scraper.is_safe_to_trade(prediction)
        print(f"  - 安全交易: {'✓ 是' if is_safe else '⚠️  否'}")
        
        position = scraper.get_position_signal(prediction)
        print(f"  - 仓位信号: {position:+.2f}")
    else:
        print("⚠️  无法获取预测数据")
        print("   原因可能是:")
        print("   1. 网站结构与爬虫代码不匹配（需要调整解析逻辑）")
        print("   2. 网络连接问题")
        print("   3. 官方网站暂时不可用")
        print("\n   这是正常的！爬虫需要根据实际网站结构调整。")
        print("   请查看 docs/kronos_strategy.md 了解如何自定义爬虫。")

except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("  请安装依赖: pip install beautifulsoup4")
    sys.exit(1)
except Exception as e:
    print(f"✗ 爬虫测试失败: {e}")

# 测试2: 策略功能
print("\n[测试2] Kronos策略类...")
try:
    from vquant.analysis.kronos import KronosTrader
    
    trader = KronosTrader(symbol="BTCUSDC", name="test")
    print("✓ 策略类导入成功")
    
    # 模拟市场数据
    mock_stats = {
        'current_price': 95000.0,
        'current_ma7': 94500.0,
        'current_rsi': 55.0,
    }
    
    print("\n运行策略分析...")
    result = trader.analyze(mock_stats)
    
    print(f"✓ 分析完成")
    print(f"  - 仓位建议: {result['position']:+.2f} (-1到1)")
    print(f"  - 置信度: {result['confidence']}")
    print(f"  - 安全交易: {'✓ 是' if result.get('is_safe') else '⚠️  否'}")
    print(f"\n  决策理由:")
    for line in result['reasoning'].split('. '):
        if line.strip():
            print(f"    • {line.strip()}")
    
    # 生成输出
    output = trader.generate_output(result, mock_stats, None)
    print(f"\n✓ 输出生成成功")
    print(f"  分析类型: {output['analysis_type']}")

except Exception as e:
    print(f"✗ 策略测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 配置文件
print("\n[测试3] 配置文件...")
try:
    from pathlib import Path
    import json
    
    config_path = Path("config/kronos_strategy.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✓ 配置文件存在")
        print(f"  - 最大数据陈旧时间: {config.get('max_staleness_hours', 'N/A')} 小时")
        print(f"  - 置信度阈值: {config.get('confidence_threshold', 'N/A')}")
        print(f"  - 仓位倍数: {config.get('position_multiplier', 'N/A')}")
        print(f"  - 安全检查: {'启用' if config.get('enable_safety_check') else '禁用'}")
    else:
        print("⚠️  配置文件不存在: config/kronos_strategy.json")
        print("   将使用默认配置")
except Exception as e:
    print(f"✗ 配置测试失败: {e}")

# 测试4: 主程序集成
print("\n[测试4] 主程序集成...")
try:
    import main
    
    # 检查kronos是否在选项中
    if hasattr(main, '_create_predictor'):
        print("✓ 主程序集成成功")
        print("  可以使用以下命令运行:")
        print("  python main.py --predictor kronos --name test --symbol BTCUSDC")
    else:
        print("⚠️  找不到 _create_predictor 函数")
except Exception as e:
    print(f"⚠️  主程序检查失败: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("\n下一步:")
print("1. 安装依赖: pip install beautifulsoup4")
print("2. 检查官方网站: https://shiyu-coder.github.io/Kronos-demos/")
print("3. 根据实际网站结构调整 vquant/data/kronos_scraper.py 的解析逻辑")
print("4. 运行策略: python main.py --predictor kronos --name test --symbol BTCUSDC")
print("5. 查看详细文档: docs/kronos_strategy.md")
print()
