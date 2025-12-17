# 图表分析工具使用说明

## 功能介绍

本工具可以自动将币安行情图表上传到ChatGPT进行分析，并生成投资建议保存为JSON文件。

## 文件说明

- `binance_chart.py`: 生成币安行情图表
- `chart_analyzer.py`: 使用Selenium将图表发送给ChatGPT分析
- `charts/`: 保存生成的图表和分析结果的目录

## 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 您还需要安装Chrome浏览器和对应版本的ChromeDriver。

### 安装ChromeDriver

#### Windows:
1. 下载ChromeDriver: https://chromedriver.chromium.org/downloads
2. 将chromedriver.exe放到系统PATH中，或与脚本同目录

#### 自动安装（推荐）:
```bash
pip install webdriver-manager
```

然后在chart_analyzer.py中使用：
```python
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

service = Service(ChromeDriverManager().install())
self.driver = webdriver.Chrome(service=service, options=options)
```

## 使用方法

### 方式一：独立运行分析器

1. 首先生成图表：
```bash
python binance_chart.py
```

2. 然后分析所有图表：
```bash
python chart_analyzer.py
```

程序会：
- 打开Chrome浏览器并导航到ChatGPT
- 提示您登录ChatGPT（如果尚未登录）
- 逐个上传charts目录中的PNG图片
- 请求ChatGPT分析每张图表
- 将分析结果保存为对应的JSON文件

### 方式二：集成到binance_chart.py

在binance_chart.py的末尾，取消注释以下代码段：

```python
print("\nAnalyzing chart with ChatGPT...")
from chart_analyzer import ChatGPTChartAnalyzer
analyzer = ChatGPTChartAnalyzer(headless=False)
analyzer.init_browser()
analyzer.navigate_to_chatgpt()
input("请确认已登录ChatGPT，然后按Enter继续...")
analyzer.analyze_chart_image(save_path, save_json=True)
analyzer.close()
print("Analysis completed!")
```

然后直接运行：
```bash
python binance_chart.py
```

## 输出格式

每张图表都会生成一个对应的JSON文件，例如：

**图表文件**: `charts/BTCUSDT_1h_20251216_205319.png`
**分析文件**: `charts/BTCUSDT_1h_20251216_205319.json`

JSON文件内容示例：
```json
{
  "image_file": "BTCUSDT_1h_20251216_205319.png",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "timestamp": "20251216_205319",
  "analysis_time": "2025-12-16 20:55:30",
  "raw_analysis": "ChatGPT的完整分析内容...",
  "structured_analysis": {
    "technical_analysis": {},
    "market_sentiment": {},
    "investment_advice": {},
    "risk_warning": ""
  }
}
```

## 自定义分析提示词

在`chart_analyzer.py`的`upload_image_and_analyze`方法中，您可以自定义prompt参数来修改发送给ChatGPT的提示词。

默认提示词包括：
- 技术分析（趋势、支撑位、阻力位、RSI、MACD、成交量）
- 市场情绪分析
- 投资建议（短期/中期、操作建议、止损/止盈位）
- 风险提示

## 注意事项

1. **登录要求**: 首次运行时需要手动登录ChatGPT，程序会保存登录状态
2. **请求频率**: 为避免被限流，程序会在每次分析之间等待10秒
3. **ChatGPT订阅**: 图片分析功能需要ChatGPT Plus订阅（GPT-4 with Vision）
4. **网络要求**: 需要能够访问ChatGPT网站
5. **选择器更新**: ChatGPT网页界面可能会更新，如果出现问题，需要更新CSS选择器

## 故障排除

### 问题1：找不到ChromeDriver
**解决方案**: 安装webdriver-manager或手动下载ChromeDriver

### 问题2：无法找到元素
**解决方案**: ChatGPT界面可能已更新，需要检查并更新CSS选择器

### 问题3：分析超时
**解决方案**: 增加`max_wait`参数的值（在`upload_image_and_analyze`方法中）

### 问题4：无法上传图片
**解决方案**: 
- 确保使用的是ChatGPT Plus账户
- 检查图片文件路径是否正确
- 尝试手动上传一次图片验证账户权限

## 高级配置

### 使用无头模式
```python
analyzer = ChatGPTChartAnalyzer(headless=True)
```

### 批量分析特定图表
```python
analyzer = ChatGPTChartAnalyzer()
analyzer.init_browser()
analyzer.navigate_to_chatgpt()
input("请确认已登录ChatGPT，然后按Enter继续...")

# 分析特定图表
for image_file in ['chart1.png', 'chart2.png']:
    analyzer.analyze_chart_image(f'charts/{image_file}', save_json=True)
    time.sleep(10)

analyzer.close()
```

## 进一步改进建议

1. **结构化解析**: 实现更智能的文本解析，将ChatGPT的回复自动解析为结构化数据
2. **错误重试**: 添加失败重试机制
3. **并行处理**: 使用多个浏览器实例加速批量处理
4. **API集成**: 如果有ChatGPT API访问权限，可以直接使用API替代Selenium
5. **定时任务**: 配合cron或Windows任务计划程序实现定时分析

## 许可证

请遵守ChatGPT的使用条款和服务协议。
