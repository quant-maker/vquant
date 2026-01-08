# Graph-Quant 项目说明

量化交易系统，基于1分钟数据架构，支持动态时间周期聚合。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 预拉取数据

```bash
# 拉取所有默认交易对的1分钟数据（从2023-01-01至今）
python -m vquant.data.manager prefetch --all --start-date 2023-01-01

# 或拉取特定交易对
python -m vquant.data.manager prefetch --symbol BTCUSDC --start-date 2023-01-01
```

### 3. 训练模型

```bash
# 使用1m数据训练1h策略（推荐）
python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 365

# 使用1m数据训练4h策略
python -m vquant.model.train --symbol ETHUSDC --interval 4h --days 365

# 使用1m数据训练日线策略
python -m vquant.model.train --symbol BNBUSDC --interval 1d --days 730
```

### 4. 查看数据

```bash
# 列出所有缓存数据
python -m vquant.data.manager list

# 查看特定数据
python -m vquant.data.manager view --symbol BTCUSDC --interval 1m --days 7
```

## 项目结构

```
graph-quant/
├── vquant/                  # 核心包
│   ├── data/               # 数据模块
│   │   ├── fetcher.py     # 数据拉取和聚合
│   │   ├── manager.py     # CLI数据管理工具
│   │   ├── prefetch.py    # 批量预拉取脚本
│   │   └── example.py     # 使用示例
│   ├── model/              # 模型模块
│   │   ├── calibrator.py  # 模型校准器
│   │   └── train.py       # 统一训练入口
│   ├── analysis/           # 策略分析
│   │   └── kelly.py       # Kelly策略
│   └── executor/           # 交易执行
├── tests/                   # 测试套件（pytest）
│   ├── test_fetcher.py    # 数据拉取测试
│   ├── test_resample.py   # 数据聚合测试
│   ├── test_prefetch.py   # 批量拉取测试
│   └── test_train.py      # 训练功能测试
├── data/                    # 数据目录
│   └── market_data.db      # SQLite数据库
├── pytest.ini              # pytest配置
└── requirements.txt        # 依赖列表
```

## 核心功能

### 数据管理

**1分钟数据架构**
- 只拉取1分钟K线数据（最细粒度）
- 在训练/回测时动态聚合为目标周期（1h/4h/1d）
- 数据存储在SQLite数据库中，支持快速查询

**优势**
- 存储效率高（只存储最细粒度）
- 灵活性强（可聚合为任意周期）
- 数据一致性好（统一数据源）

### 训练入口

统一的训练入口 `vquant.model.train`：
- 支持从1m数据动态聚合训练
- 支持多个交易对和时间周期
- 自动模型保存和特征重要性分析

```bash
python -m vquant.model.train --help
```

### 测试

使用pytest管理测试：

```bash
# 运行所有测试
pytest

# 运行快速测试（排除slow标记）
pytest -m "not slow"

# 运行特定测试
pytest tests/test_resample.py -v

# 生成覆盖率报告
pytest --cov=vquant --cov-report=html
```

测试分类：
- `@pytest.mark.unit` - 单元测试（快速）
- `@pytest.mark.integration` - 集成测试（需要缓存数据）
- `@pytest.mark.slow` - 慢速测试（涉及API请求）

## 常用命令

```bash
# 数据管理
python -m vquant.data.manager list              # 列出缓存数据
python -m vquant.data.manager prefetch --help   # 预拉取帮助
python -m vquant.data.prefetch                  # 批量预拉取

# 模型训练
python -m vquant.model.train --help            # 训练帮助
python -m vquant.model.train --symbol BTCUSDC  # 训练BTCUSDC

# 策略运行
python main.py --help                          # 查看所有策略选项
python main.py --predictor quant --name my_quant  # 量化策略
python main.py --predictor kelly --name my_kelly  # Kelly策略
python main.py --predictor kronos --name my_kronos  # Kronos策略（基于官方预测）

# 示例脚本
python -m vquant.data.example                  # resample示例

# 测试
pytest                                          # 运行所有测试
pytest -m "not slow"                           # 快速测试
pytest tests/test_resample.py -v              # 特定测试
```

## 策略说明

### Kronos策略

**基于Kronos官方预测的交易策略**

由于Kronos模型需要大量计算资源，该策略通过爬取官方网站的预测结果来生成交易信号。

特点：
- 自动爬取 https://shiyu-coder.github.io/Kronos-demos/ 的BTC/USDT预测
- **内置数据新鲜度检测**，避免官方停止更新后继续交易
- 支持配置最大数据陈旧时间（默认24小时）
- 自动阻止使用过时数据进行交易

快速开始：
```bash
# 安装依赖
pip install beautifulsoup4

# 运行策略（仅分析）
python main.py --predictor kronos --name kronos_test --symbol BTCUSDC --interval 1h

# 启用交易
python main.py --predictor kronos --name kronos_live --symbol BTCUSDC --interval 1h --trade
```

详细文档：[docs/kronos_strategy.md](docs/kronos_strategy.md)


## API限流说明

Binance API限制：
- 权重限制: 1200/分钟
- 建议延迟: ≥0.5秒/批次

如遇到限流（HTTP 429）：
```bash
python -m vquant.data.manager prefetch --all --start-date 2023-01-01 --delay 1.0
```

## 开发指南

### 添加新策略

1. 在 `vquant/analysis/` 中创建策略类
2. 实现 `_extract_features()` 方法
3. 使用 `vquant.model.train` 训练模型

### 添加新功能

1. 在相应模块中实现功能
2. 在 `tests/` 中添加测试
3. 运行 `pytest` 确保测试通过
4. 更新文档

### 代码风格

- 使用type hints
- 添加docstrings
- 遵循PEP 8规范
- 添加单元测试和集成测试

## 许可证

MIT License
