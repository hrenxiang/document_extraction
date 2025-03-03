import json
import requests
import httpx
from typing import Any, List, Optional, Union, Iterator, AsyncIterator
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk


class DeepSeekLLM(LLM):
    model: str
    """要使用的模型名称"""

    temperature: Optional[float] = None
    """模型的温度。增加温度会使模型的回答更加具有创造性。默认值：0.8"""

    stop: Optional[List[str]] = None
    """设置停止词。生成过程中，如果输出中首次出现这些停止子串，则输出会被截断"""

    tfs_z: Optional[float] = None
    """尾部自由采样，减少输出中低概率词的影响。较高的值（如 2.0）会更有效，而 1.0 禁用此设置。默认值：1"""

    top_k: Optional[int] = None
    """限制生成的词汇数量。较高的值（如 100）会生成更多元的回答，而较低的值（如 10）则会更加保守。默认值：40"""

    top_p: Optional[float] = None
    """与 top_k 配合使用。较高的值（如 0.95）会导致生成更多样的文本，而较低的值（如 0.5）会生成更专注的文本。默认值：0.9"""

    format: str = ""
    """指定输出格式（选项：json）"""

    keep_alive: Optional[Union[int, str]] = None
    """模型在内存中保持加载的时间"""

    base_url: Optional[str] = None
    """模型托管的基本 URL"""

    @property
    def _identifying_params(self) -> dict:
        """返回标识模型的参数字典"""
        return {
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """返回 LLM 的类型"""
        return "deepseek-llm"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            callbacks: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """在给定的提示文本上调用 LLM（语言模型）

        该方法会将提示传递给模型并返回生成的内容。

        参数:
            prompt: 用于生成内容的提示文本。
            stop: 生成时使用的停止词。当输出中首次出现这些停止子串时，输出将被截断。
            callbacks: 可选的回调管理器，用于处理生成过程中的事件。
            **kwargs: 其他任意关键字参数，通常会传递给模型提供者的 API 调用。

        返回:
            模型生成的文本输出，去除了提示文本。
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "top_k": self.top_k,
            "stream": False,  # 非流式调用，保持与 _stream 一致的 API 结构
            **kwargs,  # 支持传入更多自定义参数
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=headers,
                json=data,
                timeout=60  # 设置超时，防止请求挂起
            )
            response.raise_for_status()  # 如果响应状态码不是 2xx，会抛出异常

            result = response.json()
            content = result.get("response", "")

            if callbacks:
                callbacks.on_llm_new_token(content)

            # 检查停止词并截断输出
            if stop:
                for s in stop:
                    if s in content:
                        content = content.split(s)[0]
                        break

            return content

        except requests.RequestException as e:
            if callbacks:
                callbacks.on_llm_error(e)
            raise RuntimeError(f"请求失败: {e}")

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """在给定的提示文本上流式运行 LLM（语言模型）

        该方法支持流式输出，在需要逐步生成内容时使用。

        参数:
            prompt: 用于生成内容的提示文本。
            stop: 生成时使用的停止词。当输出中首次出现这些停止子串时，输出将被截断。
            run_manager: 可选的回调管理器，用于处理生成过程中的事件。
            **kwargs: 其他任意关键字参数，通常会传递给模型提供者的 API 调用。

        返回:
            一个 GenerationChunk（生成块）的迭代器，每次返回一个部分生成结果。
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "top_k": self.top_k,
            "stream": True  # 强制开启流式输出
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            headers=headers,
            json=data,
            stream=True  # 使用 stream=True 以启用流式请求
        )
        response.raise_for_status()

        # 逐行解析服务器推送事件 (SSE) 或流响应
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("response", "")
                    if run_manager:
                        run_manager.on_llm_new_token(content)

                    # 返回一个 GenerationChunk
                    yield GenerationChunk(text=content)

                    # 检查是否有停止词，并在遇到时停止流
                    if stop and any(s in content for s in stop):
                        break
                except json.JSONDecodeError as e:
                    if run_manager:
                        run_manager.on_llm_error(e)
                    continue

    async def _astream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """异步地在给定的提示文本上流式运行 LLM（语言模型）

        该方法支持异步流式输出，适用于需要逐步生成内容的场景。

        参数:
            prompt: 用于生成内容的提示文本。
            stop: 生成时使用的停止词。当输出中首次出现这些停止子串时，输出将被截断。
            run_manager: 可选的回调管理器，用于处理生成过程中的事件。
            **kwargs: 其他任意关键字参数，通常会传递给模型提供者的 API 调用。

        返回:
            一个异步的 GenerationChunk（生成块）迭代器，每次返回一个部分生成结果。
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "top_k": self.top_k,
            "stream": True
        }

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", headers=headers, json=data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("response", "")
                            if run_manager:
                                await run_manager.on_llm_new_token(content)

                            # 异步返回一个 GenerationChunk
                            yield GenerationChunk(text=content)

                            # 检查是否有停止词，并在遇到时停止流
                            if stop and any(s in content for s in stop):
                                break
                        except json.JSONDecodeError as e:
                            if run_manager:
                                await run_manager.on_llm_error(e)
                            continue
