#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import json
import base64
import requests

from datetime import datetime
from typing import Optional, Dict, Any


class PositionAdvisor:
    """
    仓位建议器 - 分析K线图并给出-1到1之间的仓位建议
    
    仓位含义:
    -1.0: 满仓做空
    -0.5: 半仓做空
     0.0: 空仓观望
     0.5: 半仓做多
     1.0: 满仓做多
    """
    
    def __init__(self, service='copilot', api_key=None, base_url=None, model=None):
        """
        初始化仓位建议器
        Args:
            service: AI服务类型 ('copilot', 'openai', 'qwen', 'deepseek')
            api_key: API密钥，如果不提供则从环境变量读取
            base_url: API基础URL（可选）
            model: 模型名称（Copilot可选: gpt-4o, claude-3.5-sonnet, o1等）
        """
        self.service = service.lower()
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or self._get_base_url()
        self.model = model or self._get_default_model()
        
        if not self.api_key:
            raise ValueError(f"API key not found. Please set it via parameter or environment variable.")
    
    def _get_api_key(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        env_vars = {
            'copilot': 'GITHUB_TOKEN',
            'openai': 'OPENAI_API_KEY',
            'qwen': 'QWEN_API_KEY',  # 改为QWEN_API_KEY以匹配.env
            'deepseek': 'DEEPSEEK_API_KEY',
        }
        env_var = env_vars.get(self.service)
        return os.getenv(env_var) if env_var else None
    
    def _get_base_url(self) -> str:
        """获取API基础URL"""
        urls = {
            'copilot': 'https://api.githubcopilot.com',
            'openai': 'https://api.openai.com/v1',
            'qwen': 'https://dashscope.aliyuncs.com/api/v1',
            'deepseek': 'https://api.deepseek.com/v1',
        }
        return urls.get(self.service, 'https://api.openai.com/v1')
    
    def _get_default_model(self) -> str:
        """获取默认模型"""
        models = {
            'copilot': 'gpt-4o',  # 也可以是 'claude-3.5-sonnet', 'o1-preview'等
            'openai': 'gpt-4o',
            'qwen': 'qwen-vl-max',
            'deepseek': 'deepseek-chat',
        }
        return models.get(self.service, 'gpt-4o')
    
    def _encode_image(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _build_analysis_prompt(self) -> str:
        """构建分析提示词"""
        return """你是一位专业的量化交易分析师。请仔细分析这张加密货币K线图，包括：

1. **趋势分析**: 当前是上涨、下跌还是盘整趋势
2. **技术指标**: 
   - MA均线走势和金叉/死叉信号
   - RSI是否超买(>70)或超卖(<30)
   - MACD的多空信号
3. **量价关系**: 成交量是否配合价格走势
4. **支撑阻力**: 关键的支撑位和阻力位
5. **市场情绪**: Buy Ratio反映的多空力量对比
6. **资金费率**: Funding Rate是否存在极端情况

基于以上分析，给出一个-1到1之间的仓位建议：
- **1.0**: 强烈看多，建议满仓做多
- **0.5到0.9**: 看多，建议半仓到大仓做多
- **0.1到0.4**: 偏多，建议小仓做多
- **-0.1到0.1**: 中性，建议空仓观望
- **-0.4到-0.1**: 偏空，建议小仓做空
- **-0.9到-0.5**: 看空，建议半仓到大仓做空
- **-1.0**: 强烈看空，建议满仓做空

请以JSON格式返回结果，格式如下：
{
    "position": 0.75,
    "confidence": "高",
    "trend": "上涨",
    "reasoning": "详细的分析理由",
    "risk_warning": "风险提示",
    "key_levels": {
        "support": 价格,
        "resistance": 价格
    }
}

只返回JSON，不要其他内容。"""
    
    def _call_copilot_api(self, image_path: str) -> Dict[str, Any]:
        """调用GitHub Copilot API"""
        base64_image = self._encode_image(image_path)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Editor-Version': 'vscode/1.95.0',
            'Editor-Plugin-Version': 'copilot-chat/0.22.0',
            'User-Agent': 'GitHubCopilotChat/0.22.0'
        }
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': self._build_analysis_prompt()
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/png;base64,{base64_image}'
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 2000,
            'temperature': 0.3,
            'stream': False
        }
        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers=headers,
            json=payload,
            timeout=90
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # 尝试解析JSON
        try:
            # 移除可能的markdown代码块标记
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果无法解析JSON，返回原始内容
            return {
                'position': 0.0,
                'confidence': '未知',
                'reasoning': content,
                'error': 'Failed to parse JSON response'
            }
    
    def _call_openai_api(self, image_path: str) -> Dict[str, Any]:
        """调用OpenAI GPT-4 Vision API"""
        base64_image = self._encode_image(image_path)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': self._build_analysis_prompt()
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/png;base64,{base64_image}'
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.3
        }
        
        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # 尝试解析JSON
        try:
            # 移除可能的markdown代码块标记
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果无法解析JSON，返回原始内容
            return {
                'position': 0.0,
                'confidence': '未知',
                'reasoning': content,
                'error': 'Failed to parse JSON response'
            }
    
    def _call_qwen_api(self, image_path: str) -> Dict[str, Any]:
        """调用通义千问Qwen-VL API"""
        # 读取图片并编码
        with open(image_path, 'rb') as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        payload = {
            'model': 'qwen-vl-max',
            'input': {
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'image': f'data:image/png;base64,{base64_image}'
                            },
                            {
                                'text': self._build_analysis_prompt()
                            }
                        ]
                    }
                ]
            },
            'parameters': {
                'result_format': 'message'
            }
        }
        
        response = requests.post(
            f'{self.base_url}/services/aigc/multimodal-generation/generation',
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['output']['choices'][0]['message']['content'][0]['text']
        
        # 尝试解析JSON
        try:
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                'position': 0.0,
                'confidence': '未知',
                'reasoning': content,
                'error': 'Failed to parse JSON response'
            }
    
    def _call_deepseek_api(self, image_path: str) -> Dict[str, Any]:
        """调用DeepSeek API (如果支持vision)"""
        base64_image = self._encode_image(image_path)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': self._build_analysis_prompt()
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/png;base64,{base64_image}'
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.3
        }
        
        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        try:
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                'position': 0.0,
                'confidence': '未知',
                'reasoning': content,
                'error': 'Failed to parse JSON response'
            }
    
    def analyze(self, image_path: str, save_json: bool = True) -> Dict[str, Any]:
        """
        分析图表并给出仓位建议
        
        Args:
            image_path: 图表图片路径
            save_json: 是否保存分析结果为JSON文件
            
        Returns:
            包含仓位建议和分析结果的字典
        """
        model_display = f"{self.service.upper()}"
        if self.service == 'copilot':
            model_display = f"GitHub Copilot ({self.model})"
        print(f"正在使用 {model_display} 分析图表...")
        # 调用对应的API
        if self.service == 'copilot':
            result = self._call_copilot_api(image_path)
        elif self.service == 'openai':
            result = self._call_openai_api(image_path)
        elif self.service == 'qwen':
            result = self._call_qwen_api(image_path)
        elif self.service == 'deepseek':
            result = self._call_deepseek_api(image_path)
        else:
            raise ValueError(f"Unsupported service: {self.service}")
        # 添加元数据
        result['timestamp'] = datetime.now().isoformat()
        result['image_path'] = image_path
        result['service'] = self.service
        result['model'] = self.model
        # 确保position在-1到1之间
        if 'position' in result:
            result['position'] = max(-1.0, min(1.0, result['position']))
        # 保存为JSON
        if save_json:
            json_path = image_path.rsplit('.', 1)[0] + '.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"分析结果已保存到: {json_path}")
        return result
    

def main():
    """
    命令行使用示例
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python position_advisor.py <image_path> [service] [model]")
        print("Services: copilot (default), openai, qwen, deepseek")
        print("Models (for copilot): gpt-4o, claude-3.5-sonnet, o1-preview, o1-mini")
        sys.exit(1)
    image_path = sys.argv[1]
    service = sys.argv[2] if len(sys.argv) > 2 else 'copilot'
    model = sys.argv[3] if len(sys.argv) > 3 else None
    # 创建分析器
    advisor = PositionAdvisor(service=service, model=model)
    
    # 分析图表
    _ = advisor.analyze(image_path, save_json=True)


if __name__ == '__main__':
    main()
