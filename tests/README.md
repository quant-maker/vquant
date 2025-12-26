# Tests

测试使用pytest框架管理。

## 安装pytest

```bash
pip install pytest pytest-cov
```

## 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_resample.py

# 运行特定测试函数
pytest tests/test_resample.py::test_resample_1h_to_4h

# 显示详细输出
pytest -v

# 显示print输出
pytest -s

# 运行并生成覆盖率报告
pytest --cov=vquant --cov-report=html

# 只运行快速测试（排除slow标记）
pytest -m "not slow"

# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration
```

## 测试分类

- `@pytest.mark.unit` - 单元测试（快速，不需要网络/数据库）
- `@pytest.mark.integration` - 集成测试（需要缓存数据）
- `@pytest.mark.slow` - 慢速测试（涉及API请求）

## 测试文件

- `test_fetcher.py` - 数据拉取功能测试
- `test_resample.py` - 数据重采样功能测试
- `test_prefetch.py` - 批量预拉取功能测试

## CI/CD

在CI环境中，可以只运行快速测试：

```bash
pytest -m "not slow" --cov=vquant --cov-report=xml
```

## 注意事项

- 集成测试需要已缓存的数据，如果没有会被跳过（skip）
- 慢速测试涉及真实API请求，请谨慎运行以避免触发限流
- 首次运行前建议先预拉取一些测试数据：
  ```bash
  python -m vquant.data.manager prefetch --symbol BTCUSDC --start-date 2025-12-25
  ```
