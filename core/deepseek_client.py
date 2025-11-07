import time
from openai import OpenAI
from .config import config

class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or config.deepseek.api_key
        if not self.api_key:
            raise ValueError("DeepSeek API密钥未提供")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=config.deepseek.base_url
        )
    
    def generate_content(self, system_prompt, user_prompt, model=None, temperature=None):
        """生成内容"""
        model = model or config.deepseek.model
        temperature = temperature or config.deepseek.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=config.deepseek.max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return None
    
    def generate_with_retry(self, system_prompt, user_prompt, max_retries=3):
        """带重试的内容生成"""
        for attempt in range(max_retries):
            content = self.generate_content(system_prompt, user_prompt)
            if content:
                return content
            print(f"生成失败，第 {attempt + 1} 次重试...")
            time.sleep(2)
        return None