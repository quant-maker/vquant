#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import json
import base64
import logging
import requests
import pandas as pd

from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from .base import BasePredictor
from vquant.model.vision import plot_candlestick


logger = logging.getLogger(__name__)


class PositionAdvisor(BasePredictor):
    """
    Position Advisor - Analyze K-line chart and provide position recommendations between -1 and 1

    Position meaning:
    -1.0: Full short position
    -0.5: Half short position
     0.0: No position (stay out)
     0.5: Half long position
     1.0: Full long position
    """

    def __init__(self, symbol: str = "BTCUSDC", name: str = "default", service="copilot", api_key=None, base_url=None, model=None):
        """
        Initialize position advisor
        Args:
            symbol: Trading symbol
            name: Strategy name
            service: AI service type ('copilot', 'openai', 'qwen', 'deepseek')
            api_key: API key, if not provided will read from environment variable
            base_url: API base URL (optional)
            model: Model name (Copilot options: gpt-4o, claude-3.5-sonnet, o1, etc.)
        """
        super().__init__(symbol, name)
        self.service = service.lower()
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or self._get_base_url()
        self.model = model or self._get_default_model()

        if not self.api_key:
            raise ValueError(
                f"API key not found. Please set it via parameter or environment variable."
            )

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variable"""
        env_vars = {
            "copilot": "GITHUB_TOKEN",
            "openai": "OPENAI_API_KEY",
            "qwen": "QWEN_API_KEY",  # Changed to QWEN_API_KEY to match .env
            "deepseek": "DEEPSEEK_API_KEY",
        }
        env_var = env_vars.get(self.service)
        return os.getenv(env_var) if env_var else None

    def _get_base_url(self) -> str:
        """获取API基础URL"""
        urls = {
            "copilot": "https://api.githubcopilot.com",
            "openai": "https://api.openai.com/v1",
            "qwen": "https://dashscope.aliyuncs.com/api/v1",
            "deepseek": "https://api.deepseek.com/v1",
        }
        return urls.get(self.service, "https://api.openai.com/v1")

    def _get_default_model(self) -> str:
        """Get default model"""
        models = {
            "copilot": "gpt-4o",  # Can also be 'claude-3.5-sonnet', 'o1-preview', etc.
            "openai": "gpt-4o",
            "qwen": "qwen-vl-max",  # Can also be 'qwen-vl-plus', 'qwen3-vl-plus', etc.
            "deepseek": "deepseek-chat",
        }
        return models.get(self.service, "gpt-4o")

    def _encode_image(self, image_path: str = None, image_bytes: bytes = None) -> str:
        """Encode image to base64"""
        if image_bytes:
            return base64.b64encode(image_bytes).decode("utf-8")
        elif image_path:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise ValueError("Either image_path or image_bytes must be provided")

    def _build_analysis_prompt(self) -> str:
        """Build analysis prompt"""
        return """You are a professional quantitative trading analyst. Please carefully analyze this cryptocurrency K-line chart, including:

1. **Trend Analysis**: Is the current trend upward, downward, or consolidating?
2. **Technical Indicators**: 
   - MA moving average trends and golden cross/death cross signals
   - Is RSI overbought (>70) or oversold (<30)?
   - MACD bullish/bearish signals
3. **Volume-Price Relationship**: Does volume confirm price movement?
4. **Support and Resistance**: Key support and resistance levels
5. **Market Sentiment**: Buy Ratio reflecting long-short power balance
6. **Funding Rate**: Are there any extreme situations in the Funding Rate?

Based on the above analysis, provide a position recommendation between -1 and 1:
- **1.0**: Strongly bullish, recommend full long position
- **0.5 to 0.9**: Bullish, recommend half to large long position
- **0.1 to 0.4**: Slightly bullish, recommend small long position
- **-0.1 to 0.1**: Neutral, recommend staying out (no position)
- **-0.4 to -0.1**: Slightly bearish, recommend small short position
- **-0.9 to -0.5**: Bearish, recommend half to large short position
- **-1.0**: Strongly bearish, recommend full short position

Please return the result in JSON format as follows:
{
    "position": 0.75,
    "confidence": "high",
    "trend": "upward",
    "reasoning": "Detailed analysis reasoning",
    "risk_warning": "Risk warning",
    "key_levels": {
        "support": price,
        "resistance": price
    }
}

Return only JSON, no other content."""

    def _call_copilot_api(
        self, image_path: str = None, image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """Call GitHub Copilot API"""
        base64_image = self._encode_image(image_path, image_bytes)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Editor-Version": "vscode/1.95.0",
            "Editor-Plugin-Version": "copilot-chat/0.22.0",
            "User-Agent": "GitHubCopilotChat/0.22.0",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_analysis_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # 尝试解析JSON
        try:
            # 移除可能的markdown代码块标记
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except json.JSONDecodeError:
            # If unable to parse JSON, return original content
            return {
                "position": 0.0,
                "confidence": "unknown",
                "reasoning": content,
                "error": "Failed to parse JSON response",
            }

    def _call_openai_api(
        self, image_path: str = None, image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """Call OpenAI GPT-4 Vision API"""
        base64_image = self._encode_image(image_path, image_bytes)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_analysis_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # 尝试解析JSON
        try:
            # 移除可能的markdown代码块标记
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except json.JSONDecodeError:
            # If unable to parse JSON, return original content
            return {
                "position": 0.0,
                "confidence": "unknown",
                "reasoning": content,
                "error": "Failed to parse JSON response",
            }

    def _call_qwen_api(
        self, image_path: str = None, image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """Call Qwen-VL API"""
        # Read and encode image
        if image_bytes:
            image_data = image_bytes
        elif image_path:
            with open(image_path, "rb") as f:
                image_data = f.read()
        else:
            raise ValueError("Either image_path or image_bytes must be provided")
        base64_image = base64.b64encode(image_data).decode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/png;base64,{base64_image}"},
                            {"text": self._build_analysis_prompt()},
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }

        response = requests.post(
            f"{self.base_url}/services/aigc/multimodal-generation/generation",
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        if response.status_code != 200:
            logger.error(f"Qwen API error response: {response.text}")
        response.raise_for_status()

        result = response.json()
        content = result["output"]["choices"][0]["message"]["content"][0]["text"]

        # Try to parse JSON
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "position": 0.0,
                "confidence": "unknown",
                "reasoning": content,
                "error": "Failed to parse JSON response",
            }

    def _call_deepseek_api(
        self, image_path: str = None, image_bytes: bytes = None, stats: dict = None
    ) -> Dict[str, Any]:
        """Call DeepSeek API (text-only, vision not supported yet)"""
        logger.warning("DeepSeek doesn't support vision models yet, using text-only analysis")

        # Build technical indicators text with data tables
        indicators_text = ""
        if stats:
            current_price = stats.get('current_price', 0)
            indicators_text = "\n\n=== Market Data ==="
            
            # Recent K-line table (no index)
            recent_klines = stats.get('recent_klines', [])
            if recent_klines:
                indicators_text += "\n\n--- Recent 24 Candlesticks ---"
                indicators_text += "\nTime         | Open    | High    | Low     | Close"
                indicators_text += "\n-------------|---------|---------|---------|--------"
                for kline in recent_klines[-24:]:
                    if 'timestamp' in kline:
                        import pandas as pd
                        ts = pd.Timestamp(kline['timestamp'])
                        time_str = ts.strftime('%Y%m%d %H:%M')
                    else:
                        time_str = 'N/A'
                    indicators_text += f"\n{time_str:12s} | {kline['Open']:7.4f} | {kline['High']:7.4f} | {kline['Low']:7.4f} | {kline['Close']:7.4f}"
            
            # Simplified Summary
            indicators_text += "\n\n--- Summary ---"
            indicators_text += f"\nCurrent Price: ${current_price:.4f}"
            
            # Add MA7, MA25, MA99 current values
            ma_dict = stats.get('ma_dict', {})
            if 7 in ma_dict:
                indicators_text += f"\nMA7: {ma_dict[7]:.4f}"
            if 25 in ma_dict:
                indicators_text += f"\nMA25: {ma_dict[25]:.4f}"
            if 99 in ma_dict:
                indicators_text += f"\nMA99: {ma_dict[99]:.4f}"
            
            indicators_text += f"\nATR: {stats.get('atr', 0):.4f} ({stats.get('atr_pct', 0):.2f}%)"
            
            # MA7 Inflection
            ma7_inflection = stats.get('ma7_inflection')
            if ma7_inflection:
                if ma7_inflection == "upward":
                    indicators_text += "\nMA7 Inflection: UPWARD (Bullish signal)"
                elif ma7_inflection == "downward":
                    indicators_text += "\nMA7 Inflection: DOWNWARD (Bearish signal)"
                else:
                    indicators_text += "\nMA7 Inflection: No change (Continuing)"
            
            indicators_text += "\n========================\n"
            indicators_text += "\nAnalyze the tables to identify trend, MA crossovers, and trade direction.\n"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        prompt = self._build_analysis_prompt() + indicators_text
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        if response.status_code != 200:
            error_detail = response.text
            logger.error(f"DeepSeek API error response: {error_detail}")
        
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "position": 0.0,
                "confidence": "unknown",
                "reasoning": content,
                "error": "Failed to parse JSON response",
            }

    def prepare_data(self, df, df_display, ma_dict, ma_dict_display, stats, args) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Prepare chart data for LLM analysis
        
        Args:
            df: Full dataframe
            df_display: Display dataframe
            ma_dict: Full MA dictionary
            ma_dict_display: Display MA dictionary
            stats: Statistics dictionary
            args: Command line arguments
            
        Returns:
            Tuple of (save_path, image_bytes)
        """
        save_path = None
        image_bytes = None
        
        if args.quiet:
            logger.info("Quiet mode: generating chart in memory (no file I/O)")
            image_bytes = plot_candlestick(
                df_display,
                symbol=self.symbol,
                save_path=None,
                ma_dict=ma_dict_display,
                stats=stats,
                return_bytes=True,
            )
        else:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"charts/{self.name}_{timestamp}.png"
            plot_candlestick(
                df_display,
                symbol=self.symbol,
                save_path=save_path,
                ma_dict=ma_dict_display,
                stats=stats,
                return_bytes=False,
            )
            logger.info(f"Chart saved: {save_path}")
        
        return save_path, image_bytes
    
    def analyze(self, stats: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze chart using LLM
        
        Args:
            stats: Statistics dictionary
            **kwargs: Additional parameters (save_path, image_bytes, args)
            
        Returns:
            Analysis result from LLM
        """
        save_path = kwargs.get('save_path')
        image_bytes = kwargs.get('image_bytes')
        args = kwargs.get('args')
        
        return self.analyze_image(
            image_path=save_path,
            image_bytes=image_bytes,
            symbol=self.symbol,
            interval=args.interval if args else None,
            current_price=stats.get('current_price'),
            stats=stats,
        )
    
    def generate_output(self, result: Dict[str, Any], stats: Dict[str, Any], args) -> Dict[str, Any]:
        """
        Generate standardized output for LLM advisor
        
        Args:
            result: Analysis result from analyze()
            stats: Statistics dictionary
            args: Command line arguments
            
        Returns:
            Standardized output dictionary
        """
        result['analysis_type'] = 'advisor'
        return result
    
    def analyze_image(
        self,
        image_path: str = None,
        image_bytes: bytes = None,
        symbol: str = None,
        interval: str = None,
        current_price: float = None,
        stats: dict = None,
    ) -> Dict[str, Any]:
        """
        Analyze chart and provide position recommendation

        Args:
            image_path: Chart image path (optional if image_bytes provided)
            image_bytes: Chart image bytes (optional if image_path provided)
            symbol: Trading pair symbol (e.g., BTCUSDT)
            interval: Time interval (e.g., 1h, 4h, 1d)
            current_price: Current market price
            stats: Technical indicators and market stats (for text-only models)

        Returns:
            Dictionary containing position recommendation and analysis results
        """
        if image_path is None and image_bytes is None:
            raise ValueError("Either image_path or image_bytes must be provided")
        model_display = f"{self.service.upper()}"
        if self.service == "copilot":
            model_display = f"GitHub Copilot ({self.model})"

        source_info = "memory" if image_bytes else image_path
        logger.info(f"Analyzing chart with {model_display}: {source_info}")

        # Call corresponding API
        if self.service == "copilot":
            result = self._call_copilot_api(image_path, image_bytes)
        elif self.service == "openai":
            result = self._call_openai_api(image_path, image_bytes)
        elif self.service == "qwen":
            result = self._call_qwen_api(image_path, image_bytes)
        elif self.service == "deepseek":
            result = self._call_deepseek_api(image_path, image_bytes, stats)
        else:
            raise ValueError(f"Unsupported service: {self.service}")

        # Add metadata
        result["timestamp"] = datetime.now().isoformat()
        result["image_path"] = image_path if image_path else "memory"
        result["service"] = self.service
        result["model"] = self.model

        # Add trading information for Trader
        if symbol:
            result["symbol"] = symbol
        if current_price is not None:
            result["current_price"] = current_price

        # Ensure position is between -1 and 1
        if "position" in result:
            result["position"] = max(-1.0, min(1.0, result["position"]))

        # Save as JSON
        if image_path:
            json_path = image_path.rsplit(".", 1)[0] + ".json"
        else:
            # 如果没有图片路径（quiet模式），使用和图片相同的命名规则
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol_str = symbol if symbol else "unknown"
            interval_str = interval if interval else "unknown"
            json_path = f"charts/{symbol_str}_{interval_str}_{timestamp}.json"
            os.makedirs("charts", exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Analysis result saved to: {json_path}")
        logger.debug(
            f"Analysis result: position={result.get('position')}, confidence={result.get('confidence')}"
        )

        return result


def main():
    """
    命令行使用示例
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python position_advisor.py <image_path> [service] [model]")
        print("Services: copilot (default), openai, qwen, deepseek")
        print("Models:")
        print("  Copilot: gpt-4o, claude-3.5-sonnet, o1-preview, o1-mini")
        print("  Qwen: qwen-vl-max, qwen-vl-plus, qwen3-vl-plus")
        sys.exit(1)
    image_path = sys.argv[1]
    service = sys.argv[2] if len(sys.argv) > 2 else "copilot"
    model = sys.argv[3] if len(sys.argv) > 3 else None
    # 创建分析器
    advisor = PositionAdvisor(symbol="BTCUSDC", service=service, model=model)

    # 分析图表
    _ = advisor.analyze_image(image_path)


if __name__ == "__main__":
    main()
