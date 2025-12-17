import os
import json
import base64
from datetime import datetime
import requests
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()


class ChartAnalyzerAPI:
    """使用国内AI服务API分析行情图表"""

    def __init__(self, service="qwen", api_key=None):
        """
        初始化分析器

        参数:
            service: 使用的服务 ('qwen', 'glm', 'wenxin', 'openai')
            api_key: API密钥
        """
        self.service = service
        self.api_key = api_key or os.getenv(f"{service.upper()}_API_KEY")

        if not self.api_key:
            print(f"警告: 未设置 {service.upper()}_API_KEY")

    def encode_image_to_base64(self, image_path):
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_with_qwen(self, image_path, prompt):
        """
        使用阿里云通义千问VL分析图片
        文档: https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-api
        """
        try:
            import dashscope
            from dashscope import MultiModalConversation

            dashscope.api_key = self.api_key

            # 读取图片并转换为URL格式（本地文件）
            image_path_abs = os.path.abspath(image_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"file://{image_path_abs}"},
                        {"text": prompt},
                    ],
                }
            ]

            response = MultiModalConversation.call(
                model="qwen-vl-plus", messages=messages  # 或 qwen-vl-max
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content[0]["text"]
            else:
                print(f"通义千问API错误: {response.code} - {response.message}")
                return None

        except Exception as e:
            print(f"通义千问分析失败: {e}")
            return None

    def analyze_with_glm(self, image_path, prompt):
        """
        使用智谱GLM-4V分析图片
        文档: https://open.bigmodel.cn/dev/api#glm-4v
        """
        try:
            image_base64 = self.encode_image_to_base64(image_path)

            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            data = {
                "model": "glm-4v",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }

            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"智谱GLM-4V分析失败: {e}")
            return None

    def analyze_with_wenxin(self, image_path, prompt):
        """
        使用百度文心一言4.0分析图片
        文档: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t
        """
        try:
            # 首先获取access_token
            access_token = self._get_wenxin_access_token()
            if not access_token:
                return None

            image_base64 = self.encode_image_to_base64(image_path)

            url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={access_token}"

            headers = {"Content-Type": "application/json"}

            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_base64},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result["result"]

        except Exception as e:
            print(f"文心一言分析失败: {e}")
            return None

    def _get_wenxin_access_token(self):
        """获取文心一言的access_token"""
        try:
            # API Key和Secret Key从环境变量获取
            api_key = os.getenv("WENXIN_API_KEY")
            secret_key = os.getenv("WENXIN_SECRET_KEY")

            if not api_key or not secret_key:
                print("请设置 WENXIN_API_KEY 和 WENXIN_SECRET_KEY 环境变量")
                return None

            url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"

            response = requests.post(url, timeout=10)
            response.raise_for_status()

            return response.json().get("access_token")

        except Exception as e:
            print(f"获取access_token失败: {e}")
            return None

    def analyze_with_openai(self, image_path, prompt):
        """
        使用OpenAI GPT-4 Vision分析图片（需要API key和可访问的网络）
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            image_base64 = self.encode_image_to_base64(image_path)

            response = client.chat.completions.create(
                model="gpt-4o",  # 或 gpt-4-vision-preview
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI分析失败: {e}")
            return None

    def get_analysis_prompt(self):
        """获取分析提示词"""
        return """请分析这张加密货币行情图表，综合考虑以下因素：

1. 价格趋势（K线形态、移动平均线MA7/MA25/MA99）
2. 技术指标（RSI超买超卖、MACD金叉死叉）
3. 成交量与价格配合
4. 买卖比例和资金费率
5. 支撑位和阻力位

基于以上分析，给出一个仓位建议值pos（范围-1到1）：
- pos = 1：满仓做多（强烈看涨）
- pos = 0.5：半仓做多（偏多）
- pos = 0：空仓观望（中性）
- pos = -0.5：半仓做空（偏空）
- pos = -1：满仓做空（强烈看跌）

请严格按以下格式输出（只输出这3行）：
pos: [数值]
reason: [一句话说明理由，不超过50字]
risk: [主要风险点，不超过30字]

示例：
pos: 0.3
reason: 价格企稳反弹，MACD金叉，但成交量不足，建议小仓位做多
risk: 若跌破85000支撑位需立即止损"""

    def analyze_chart(self, image_path, prompt=None):
        """
        分析图表图片

        参数:
            image_path: 图片路径
            prompt: 自定义提示词

        返回:
            分析结果文本
        """
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return None

        if prompt is None:
            prompt = self.get_analysis_prompt()

        print(f"使用 {self.service} 服务分析图片...")

        if self.service == "qwen":
            return self.analyze_with_qwen(image_path, prompt)
        elif self.service == "glm":
            return self.analyze_with_glm(image_path, prompt)
        elif self.service == "wenxin":
            return self.analyze_with_wenxin(image_path, prompt)
        elif self.service == "openai":
            return self.analyze_with_openai(image_path, prompt)
        else:
            print(f"不支持的服务: {self.service}")
            return None

    def parse_analysis_to_json(self, analysis_text, image_filename):
        """
        将分析结果解析为结构化的JSON

        参数:
            analysis_text: 分析回复文本
            image_filename: 对应的图片文件名

        返回:
            包含分析结果的字典
        """
        parts = image_filename.replace(".png", "").split("_")

        # 提取action信息
        action = self._extract_action(analysis_text)

        result = {
            "image_file": image_filename,
            "symbol": parts[0] if len(parts) > 0 else "UNKNOWN",
            "interval": parts[1] if len(parts) > 1 else "UNKNOWN",
            "timestamp": (
                f"{parts[2]}_{parts[3]}"
                if len(parts) > 3
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
        }

        return result

    def _extract_action(self, text):
        """从分析文本中提取action信息"""
        import re

        action = {"pos": 0.0, "reason": "", "risk": ""}

        # 提取pos值
        pos_match = re.search(r"pos[:\s]+(-?[0-9.]+)", text, re.IGNORECASE)
        if pos_match:
            try:
                pos_value = float(pos_match.group(1))
                # 确保pos在-1到1之间
                action["pos"] = max(-1.0, min(1.0, pos_value))
            except:
                pass

        # 提取reason
        reason_match = re.search(r"reason[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
        if reason_match:
            action["reason"] = reason_match.group(1).strip()

        # 提取risk
        risk_match = re.search(r"risk[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
        if risk_match:
            action["risk"] = risk_match.group(1).strip()

        return action

    def save_analysis_to_json(self, analysis_data, output_path):
        """保存分析结果到JSON文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        print(f"分析结果已保存: {output_path}")

    def analyze_chart_and_save(self, image_path, save_json=True):
        """
        分析单张图表并保存结果

        参数:
            image_path: 图片路径
            save_json: 是否保存JSON结果

        返回:
            分析结果字典
        """
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return None

        # 分析图片
        analysis_text = self.analyze_chart(image_path)

        if analysis_text is None:
            print("分析失败")
            return None

        # 解析为JSON
        image_filename = os.path.basename(image_path)
        analysis_data = self.parse_analysis_to_json(analysis_text, image_filename)

        # 保存JSON
        if save_json:
            json_filename = image_filename.replace(".png", ".json")
            json_path = os.path.join(
                os.path.dirname(image_path) or "charts", json_filename
            )
            self.save_analysis_to_json(analysis_data, json_path)

        return analysis_data

    def analyze_all_charts(self, charts_dir="charts"):
        """
        分析指定目录中的所有图表

        参数:
            charts_dir: 图表目录路径
        """
        if not os.path.exists(charts_dir):
            print(f"目录不存在: {charts_dir}")
            return

        # 获取所有PNG图片
        all_png_files = [f for f in os.listdir(charts_dir) if f.endswith(".png")]

        if not all_png_files:
            print("未找到图表图片")
            return

        # 只保留没有对应JSON文件的PNG
        image_files = []
        for png_file in all_png_files:
            json_file = png_file.replace(".png", ".json")
            json_path = os.path.join(charts_dir, json_file)
            if not os.path.exists(json_path):
                image_files.append(png_file)

        print(f"找到 {len(all_png_files)} 张图表，其中 {len(image_files)} 张需要分析")

        if not image_files:
            print("所有图表都已分析完成")
            return []

        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 分析 {image_file}...")
            image_path = os.path.join(charts_dir, image_file)

            result = self.analyze_chart_and_save(image_path, save_json=True)

            if result:
                results.append(result)
                print(f"✓ 分析完成")
            else:
                print(f"✗ 分析失败")

            # 等待避免API限流
            if i < len(image_files):
                import time

                print("等待3秒...")
                time.sleep(3)

        print(f"\n分析完成！共处理 {len(results)} 张图片")
        return results


def setup_api_key():
    """引导用户设置API密钥"""
    print("=" * 60)
    print("图表分析器 - API密钥设置")
    print("=" * 60)
    print("\n请选择要使用的AI服务：")
    print("1. 通义千问VL (阿里云) - 推荐")
    print("2. 智谱GLM-4V")
    print("3. 文心一言4.0 (百度)")
    print("4. OpenAI GPT-4 Vision (需要国际网络)")

    choice = input("\n请输入选项 (1-4): ").strip()

    service_map = {
        "1": ("qwen", "QWEN_API_KEY", "https://dashscope.console.aliyun.com/apiKey"),
        "2": ("glm", "GLM_API_KEY", "https://open.bigmodel.cn/usercenter/apikeys"),
        "3": (
            "wenxin",
            "WENXIN_API_KEY",
            "https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application",
        ),
        "4": ("openai", "OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    }

    if choice not in service_map:
        print("无效选项")
        return

    service, env_key, url = service_map[choice]

    print(f"\n选择了: {service}")
    print(f"请访问: {url}")
    print("获取API密钥后，有两种设置方式：")
    print(f"\n方式1 - 设置环境变量（推荐）:")
    print(f"  Windows PowerShell:")
    print(f"    $env:{env_key}='your-api-key-here'")
    print(f"  Linux/Mac:")
    print(f"    export {env_key}='your-api-key-here'")
    print(f"\n方式2 - 创建.env文件:")
    print(f"  在项目目录创建.env文件，内容:")
    print(f"    {env_key}=your-api-key-here")

    if service == "wenxin":
        print(f"  注意：文心一言需要两个密钥:")
        print(f"    WENXIN_API_KEY=your-api-key")
        print(f"    WENXIN_SECRET_KEY=your-secret-key")


if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_api_key()
        sys.exit(0)

    # 尝试加载.env文件
    try:
        from dotenv import load_dotenv

        load_dotenv()
        print("已加载.env配置")
    except ImportError:
        print("提示: 安装python-dotenv可以使用.env文件管理API密钥")
        print("  pip install python-dotenv")

    # 检测可用的API密钥
    available_services = []
    if os.getenv("QWEN_API_KEY"):
        available_services.append("qwen")
    if os.getenv("GLM_API_KEY"):
        available_services.append("glm")
    if os.getenv("WENXIN_API_KEY") and os.getenv("WENXIN_SECRET_KEY"):
        available_services.append("wenxin")
    if os.getenv("OPENAI_API_KEY"):
        available_services.append("openai")

    if not available_services:
        print("未检测到API密钥！")
        print("运行以下命令查看设置说明:")
        print("  python chart_analyzer_api.py setup")
        sys.exit(1)

    # 使用第一个可用的服务
    service = available_services[0]
    print(f"使用服务: {service}")

    # 创建分析器并分析所有图表
    analyzer = ChartAnalyzerAPI(service=service)
    analyzer.analyze_all_charts("charts")
