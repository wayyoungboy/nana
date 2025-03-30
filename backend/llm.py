import httpx
import asyncio
from typing import List, Dict
import json
import re

from openai import OpenAI


class LLMService:
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url

    async def generate_response(
            self,
            message: str,
            temperature: float = 0.7,
            max_retries: int = 3,
            is_json: bool = True
    ) -> str:
        retry_count = 0

        while retry_count < max_retries:
            try:
                client = OpenAI(
                    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                completion = client.chat.completions.create(
                    model="qwen2.5-32b-instruct",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                    messages=[
                        {'role': 'user', 'content': message}
                    ]

                )
                raw_response = completion.choices[0].message.content

                print("raw_response:", raw_response)
                if is_json:
                    return self._parse_json_response(raw_response)
                else:
                    return raw_response

            except Exception as e:
                retry_count += 1
                print(f"LLM Error (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    await asyncio.sleep(1)

    @staticmethod
    def _parse_json_response(raw_response: str) -> Dict:
        # 尝试直接解析
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            # 使用正则表达式匹配 ```json 或 ``` 包裹的内容
            try:
                # 先尝试匹配 ```json 格式
                pattern = r'```(?:json\n|\n)?([^`]*?)```'
                matches = re.search(pattern, raw_response, re.DOTALL)

                if matches:
                    json_str = matches.group(1).strip()
                    return json.loads(json_str)
                raise ValueError("No valid JSON block found")
            except Exception as e:
                raise ValueError(f"Failed to parse JSON response: {str(e)}")
