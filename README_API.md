# 图表分析工具使用说明（API版本）

## 功能介绍

使用国内AI服务API分析币安行情图表，生成投资建议并保存为JSON文件。

**✅ 优势：**
- 无需VPN或国际支付
- 稳定可靠，速度快
- API调用，无需浏览器自动化
- 支持多个国内AI服务

## 支持的AI服务

| 服务 | 优点 | 费用 | 获取API密钥 |
|------|------|------|------------|
| **通义千问VL** (推荐) | 阿里云服务，稳定快速 | 0.008元/次 | [申请地址](https://dashscope.console.aliyun.com/apiKey) |
| **智谱GLM-4V** | 清华出品，效果好 | 0.05元/次 | [申请地址](https://open.bigmodel.cn/usercenter/apikeys) |
| **文心一言4.0** | 百度服务 | 按tokens计费 | [申请地址](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application) |
| OpenAI GPT-4V | 效果最好（需国际网络） | $0.01/次 | [申请地址](https://platform.openai.com/api-keys) |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 获取API密钥

运行设置向导：
```bash
python chart_analyzer_api.py setup
```

#### 通义千问（推荐）
1. 访问 https://dashscope.console.aliyun.com/apiKey
2. 登录阿里云账号（可用支付宝登录）
3. 创建API-KEY
4. 首次开通送免费额度

#### 智谱GLM-4V
1. 访问 https://open.bigmodel.cn/usercenter/apikeys
2. 注册/登录账号
3. 生成API Key
4. 新用户有免费tokens

#### 文心一言
1. 访问百度智能云千帆平台
2. 创建应用获取API Key和Secret Key

### 3. 设置API密钥

**方式一：.env文件（推荐）**

复制示例文件并配置：
```bash
cp .env.example .env
```

然后编辑`.env`文件，填入你的API密钥：
```
QWEN_API_KEY=your-qwen-api-key-here
GLM_API_KEY=your-glm-api-key-here
WENXIN_API_KEY=your-wenxin-api-key
WENXIN_SECRET_KEY=your-wenxin-secret-key
OPENAI_API_KEY=your-openai-api-key
```

**方式二：环境变量**

Windows PowerShell:
```powershell
$env:QWEN_API_KEY='your-api-key-here'
```

Git Bash:
```bash
export QWEN_API_KEY='your-api-key-here'
```

### 4. 运行分析

```bash
python chart_analyzer_api.py
```

程序会自动：
- 检测可用的API密钥
- 分析charts目录中的所有PNG图片
- 为每张图片生成对应的JSON分析结果

## 使用示例

### 独立运行

```python
from chart_analyzer_api import ChartAnalyzerAPI

# 创建分析器（自动检测API密钥）
analyzer = ChartAnalyzerAPI(service='qwen')

# 分析单张图片
result = analyzer.analyze_chart_and_save('charts/BTCUSDT_1h_20251216_205319.png')

# 批量分析
analyzer.analyze_all_charts('charts')
```

### 集成到binance_chart.py

在[binance_chart.py](binance_chart.py)末尾取消注释：

```python
print("\nAnalyzing chart with AI...")
from chart_analyzer_api import ChartAnalyzerAPI
analyzer = ChartAnalyzerAPI(service='qwen')
analyzer.analyze_chart_and_save(save_path, save_json=True)
print("Analysis completed!")
```

然后直接运行：
```bash
python binance_chart.py
```

### 指定服务

```python
# 使用通义千问
analyzer = ChartAnalyzerAPI(service='qwen', api_key='your-key')

# 使用智谱GLM-4V
analyzer = ChartAnalyzerAPI(service='glm', api_key='your-key')

# 使用文心一言
analyzer = ChartAnalyzerAPI(service='wenxin')  # 从环境变量读取

# 使用OpenAI
analyzer = ChartAnalyzerAPI(service='openai', api_key='your-key')
```

## 输出格式

每张图表生成对应的JSON文件：

**图表**: `charts/BTCUSDT_1h_20251216_205319.png`  
**分析**: `charts/BTCUSDT_1h_20251216_205319.json`

JSON内容示例：
```json
{
  "image_file": "BTCUSDT_1h_20251216_205319.png",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "timestamp": "20251216_205319",
  "analysis_time": "2025-12-16 21:30:45",
  "service": "qwen",
  "raw_analysis": "详细的AI分析内容...",
  "structured_analysis": {
    "trend": "上涨",
    "support_levels": [95000, 94500],
    "resistance_levels": [97000, 98000],
    "rsi_status": "中性",
    "macd_status": "多头",
    "recommendation": "短期持有",
    "stop_loss": "94500",
    "take_profit": "97500",
    "risk_warning": "..."
  }
}
```

## 成本估算

以通义千问为例：
- 单次分析约0.008元
- 分析100张图表约0.8元
- 每日生成1张图表，月成本约0.24元

## 常见问题

### Q: 如何获得免费额度？
**A:** 
- 通义千问：新用户首次开通送100万tokens
- 智谱GLM：新用户送免费tokens
- 可以先用免费额度测试

### Q: API密钥如何充值？
**A:** 
- 通义千问：在阿里云控制台充值，支持支付宝/银行卡
- 智谱GLM：在官网充值，支持微信/支付宝
- 文心一言：百度智能云充值

### Q: 分析速度如何？
**A:** 单张图片分析通常3-10秒，比Selenium快很多

### Q: 支持哪些图片格式？
**A:** 支持PNG、JPG、JPEG等常见格式

### Q: API调用失败怎么办？
**A:** 
1. 检查API密钥是否正确
2. 检查账户余额是否充足
3. 检查网络连接
4. 查看错误日志

### Q: 如何提高分析质量？
**A:** 
- 使用更高级的模型（qwen-vl-max, glm-4v）
- 优化提示词（在代码中修改`get_analysis_prompt`方法）
- 确保图表清晰、信息完整

## 高级配置

### 自定义提示词

```python
analyzer = ChartAnalyzerAPI(service='qwen')

custom_prompt = """
请分析图表并重点关注：
1. 短期（1-3天）的价格走势
2. 关键支撑和阻力位
3. 具体的买卖建议和价位
"""

result = analyzer.analyze_chart('chart.png', prompt=custom_prompt)
```

### 批量处理特定图表

```python
import os
import glob

analyzer = ChartAnalyzerAPI(service='qwen')

# 只分析今天的图表
today = '20251216'
for image_path in glob.glob(f'charts/*{today}*.png'):
    analyzer.analyze_chart_and_save(image_path)
```

### 定时任务

Windows任务计划程序：
```powershell
# 创建每小时运行的任务
$action = New-ScheduledTaskAction -Execute "python" -Argument "C:\path\to\binance_chart.py"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "BinanceChartAnalysis" -Action $action -Trigger $trigger
```

Linux Cron:
```bash
# 每小时运行
0 * * * * cd /path/to/project && python binance_chart.py
```

## 性能优化

### 并行处理（高级）

```python
from concurrent.futures import ThreadPoolExecutor
from chart_analyzer_api import ChartAnalyzerAPI

def analyze_single(image_file):
    analyzer = ChartAnalyzerAPI(service='qwen')
    return analyzer.analyze_chart_and_save(f'charts/{image_file}')

image_files = [f for f in os.listdir('charts') if f.endswith('.png')]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(analyze_single, image_files))
```

## 服务对比

| 特性 | 通义千问VL | 智谱GLM-4V | 文心一言 |
|------|-----------|-----------|---------|
| 响应速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 分析质量 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 价格 | 最便宜 | 中等 | 中等 |
| 稳定性 | 极高 | 高 | 高 |
| 免费额度 | 100万tokens | 有 | 有 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 总结

相比Selenium方案，API方案具有：
- ✅ 无需浏览器，更稳定
- ✅ 速度更快（3-10秒 vs 30-60秒）
- ✅ 成本更低（几分钱 vs ChatGPT Plus $20/月）
- ✅ 在中国大陆可直接使用
- ✅ 更易于自动化和集成

**推荐使用通义千问VL作为首选方案！**
