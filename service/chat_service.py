import logging
import os
from typing import Optional, Any

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from core.custom_llm import DeepSeekLLM
from service.chat_history import chat_with_history_stream
from service.retrieval_chain import initialize_retrieval_chain
from service.vector_store import vector_store

os.environ["OMP_NUM_THREADS"] = "1"

# 配置日志记录
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def initialize_simple_chain(llm):
    """创建一个没有combine_docs_chain的简单检索链"""

    # 提示模板，仅使用检索结果和用户输入
    prompt_template = ChatPromptTemplate.from_messages([
        (
            'system',
            """
            你是一个用于问答任务的助手。 \
            如果我问你关于上下文以及历史对话的信息，\  
            你可以从聊天历史中收集信息来回答。\  
            如果你不知道答案，请直接说你不知道。\  
            保持回答简洁，回答时必须使用markdown格式。
            """
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}')
    ])

    return prompt_template | llm


def _prepare_chat_runnable(file_path: Optional[str], user_id: str, session_id: str) -> Any:
    """
    处理向量存储、文档加载及检索链（或备用 prompt）的初始化，
    返回用于对话的 chat_runnable 对象
    """
    try:
        # 2. 初始化 LLM
        model = DeepSeekLLM(
            model="deepseek-r1:7b",
            base_url="http://192.168.64.1:11434"
        )

        # 3. 创建检索链（如果向量存储存在），否则构造默认的 prompt 链
        logger.info("创建检索链")
        retrieval_chain = initialize_retrieval_chain(vector_store, model, 3, file_path, user_id, session_id)
        return retrieval_chain

    except Exception as e:
        logger.exception("准备聊天链时出错：%s", e)
        raise


def generate_stream(
        file_path: Optional[str],
        user_input: str,
        user_id: str,
        session_id: str,
) -> Any:
    """
    主流程：流式输出
    """
    try:
        chat_runnable = _prepare_chat_runnable(file_path=file_path, user_id=user_id, session_id=session_id)
        logger.info("生成流式输出")
        yield from chat_with_history_stream(chat_runnable, user_input, session_id, "retrieval")
    except Exception as e:
        logger.exception("生成流式结果时出错：%s", e)
        raise
