# 快速设置指南

## 1. 配置API密钥

复制示例文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的API密钥。

## 2. 使用Git Bash运行命令

推荐使用Git Bash（比PowerShell更友好）：

### 安装依赖
```bash
pip install -r requirements.txt
```

### 生成图表
```bash
python binance_chart.py
```

### 分析图表
```bash
python chart_analyzer_api.py
```

### 设置环境变量（临时）
```bash
export QWEN_API_KEY='your-api-key-here'
```

## 3. 测试
```bash
# 测试API连接
python -c "from chart_analyzer_api import ChartAnalyzerAPI; print('✓ API配置成功')"
```

## 常见问题

**Q: .env文件在哪里？**
A: 在项目根目录（与README.md同级），如果不存在请复制.env.example

**Q: 如何检查API key是否加载？**
A: 在Git Bash中运行：
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('QWEN_API_KEY'))"
```

**Q: 支持哪些AI服务？**
A: 通义千问(qwen)、智谱GLM(glm)、文心一言(wenxin)、OpenAI(openai)
