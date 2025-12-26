# 项目重构总结

## 文件迁移

### 数据相关文件 → `vquant/data/`

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `data_manager.py` | `vquant/data/manager.py` | CLI数据管理工具 |
| `prefetch_data.py` | `vquant/data/prefetch.py` | 批量预拉取脚本 |
| `example_resample.py` | `vquant/data/example.py` | resample使用示例 |

### 训练相关文件 → `vquant/model/`

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `train_kelly.py` | `vquant/model/train.py` | 统一训练入口 |

### 测试文件 → `tests/`

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `test_resample.py` | `tests/test_resample.py` | resample功能测试 |
| `test_prefetch.py` | `tests/test_prefetch.py` | prefetch功能测试 |
| ➕ | `tests/test_fetcher.py` | 数据拉取测试（新增） |
| ➕ | `tests/test_train.py` | 训练功能测试（新增） |

## 命令更新

### 数据管理

```bash
# 旧命令
python data_manager.py list
python data_manager.py prefetch --symbol BTCUSDC --days 365

# 新命令
python -m vquant.data.manager list
python -m vquant.data.manager prefetch --symbol BTCUSDC --start-date 2023-01-01
```

### 训练模型

```bash
# 旧命令
python train_kelly.py --symbol BTCUSDC --interval 1h --days 180

# 新命令（推荐：使用1m数据）
python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 365

# 新命令（兼容模式：直接使用interval数据）
python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 180 --no-use-1m
```

### 测试

```bash
# 旧命令
python test_resample.py

# 新命令
pytest tests/test_resample.py -v
pytest                          # 运行所有测试
pytest -m "not slow"           # 只运行快速测试
```

## 架构升级

### 1. 1分钟数据架构

**变更**
- 从拉取多种interval（1h/4h/1d）改为只拉取1m数据
- 在训练/回测时通过 `resample_klines()` 动态聚合

**优势**
- ✅ 存储效率：只存储最细粒度数据
- ✅ 灵活性：可聚合为任意周期
- ✅ 数据一致性：统一数据源
- ✅ 流量控制：减少API请求次数

**参数变化**
- `--days` 参数改为 `--start-date`（YYYY-MM-DD格式）
- 添加 `--delay` 参数控制请求延迟（默认0.5秒）

### 2. 统一训练入口

**变更**
- `train_kelly.py` → `vquant.model.train`
- 支持从1m数据动态聚合训练（默认启用）
- 添加 `--no-use-1m` 参数兼容旧模式

**新功能**
- ✅ 自动从1m数据聚合为目标interval
- ✅ 更好的日志输出（显示步骤编号）
- ✅ 更详细的帮助文档

**使用方式**
```bash
# 推荐：使用1m数据训练（需先prefetch 1m数据）
python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 365

# 兼容：直接使用缓存的interval数据
python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 180 --no-use-1m
```

### 3. Pytest测试框架

**变更**
- 从独立脚本改为pytest测试套件
- 添加测试标记（unit/integration/slow）
- 添加pytest.ini配置

**测试分类**
- `unit`: 单元测试，快速，不需要外部资源
- `integration`: 集成测试，需要缓存数据
- `slow`: 慢速测试，涉及API请求

**运行方式**
```bash
pytest                    # 所有测试
pytest -m unit           # 只运行单元测试
pytest -m "not slow"     # 排除慢速测试
pytest -v                # 详细输出
pytest --cov=vquant      # 生成覆盖率报告
```

## 迁移指南

### 对于现有脚本

如果你有使用旧命令的脚本，需要更新：

1. **数据管理脚本**
   ```bash
   # 旧
   python data_manager.py prefetch --symbol BTCUSDC --interval 1h --days 365
   
   # 新（只拉1m数据）
   python -m vquant.data.manager prefetch --symbol BTCUSDC --start-date 2023-01-01
   ```

2. **训练脚本**
   ```bash
   # 旧
   python train_kelly.py --symbol BTCUSDC --interval 1h --days 180
   
   # 新（推荐）
   python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 365
   ```

3. **测试脚本**
   ```bash
   # 旧
   python test_resample.py
   
   # 新
   pytest tests/test_resample.py -v
   ```

### 对于Python代码

如果你的代码中直接导入了旧模块，需要更新import：

```python
# 数据相关
from vquant.data import get_cached_klines, resample_klines, prefetch_all_data

# 训练相关
from vquant.model.train import train_kelly_model

# 使用resample聚合数据
df_1m = get_cached_klines('BTCUSDC', '1m', days=365)
df_1h = resample_klines(df_1m, '1h')
```

## 新增文件

- ✅ `README.md` - 项目总览文档
- ✅ `pytest.ini` - pytest配置文件
- ✅ `tests/__init__.py` - 测试包初始化
- ✅ `tests/README.md` - 测试使用说明
- ✅ `tests/test_fetcher.py` - 数据拉取测试
- ✅ `tests/test_train.py` - 训练功能测试
- ✅ `MIGRATION.md` - 本迁移指南

## 待清理文件

以下文件已迁移，可以删除（如果还存在）：
- ❌ `data_manager.py`
- ❌ `prefetch_data.py`
- ❌ `example_resample.py`
- ❌ `test_resample.py` (根目录)
- ❌ `test_prefetch.py` (根目录)
- ❌ `train_kelly.py`

## 更新日志

### 2025-12-26

**数据架构升级**
- 改为只拉取1m数据架构
- 添加 `resample_klines()` 函数支持动态聚合
- 更新参数：`--days` → `--start-date`

**文件重组**
- 数据相关文件移至 `vquant/data/`
- 训练文件移至 `vquant/model/`
- 测试文件移至 `tests/`，使用pytest管理

**功能增强**
- 统一训练入口 `vquant.model.train`
- 添加流量控制机制（防API限流）
- 完善测试覆盖（4个测试模块，13+个测试）

**文档完善**
- 添加 `README.md` 项目总览
- 添加 `tests/README.md` 测试说明
- 更新所有命令示例

## 总结

本次重构实现了：
1. ✅ **模块化**: 按功能组织文件（data/model/tests）
2. ✅ **标准化**: 文件名不超过一个单词
3. ✅ **测试化**: 使用pytest管理测试
4. ✅ **高效化**: 1m数据架构提升存储和灵活性
5. ✅ **文档化**: 完整的README和迁移指南

项目现在结构清晰，易于维护和扩展！
