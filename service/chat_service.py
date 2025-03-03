import logging
import os
from typing import Dict
from typing import Optional, Any

from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import MessagesPlaceholder

from core.custom_llm import DeepSeekLLM
from service.chat_history import chat_with_history, chat_with_history_stream
from service.document_processor import load_and_split_document
from service.retrieval_chain import initialize_retrieval_chain
from service.vector_store import initialize_vector_store

os.environ["OMP_NUM_THREADS"] = "1"

# 配置日志记录
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 全局向量存储
session_vector_stores: Dict[str, Chroma] = {}


def clean_session_vector_stores(session_id: str):
    session_vector_stores.pop(session_id, None)


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


def _prepare_chat_runnable(file_path: Optional[str], session_id: str) -> Any:
    """
    处理向量存储、文档加载及检索链（或备用 prompt）的初始化，
    返回用于对话的 chat_runnable 对象
    """
    try:
        # 1. 获取或初始化向量存储
        vector_store = session_vector_stores.get(session_id)
        if file_path:
            logger.info("加载并拆分文档：%s", file_path)
            text_chunks = load_and_split_document(file_path)
            if vector_store is None:
                logger.info("向量存储不存在，初始化向量存储")
                vector_store = initialize_vector_store(text_chunks)
                session_vector_stores[session_id] = vector_store
            else:
                logger.info("向量存储已存在，更新文档")
                docs = filter_complex_metadata(text_chunks)
                if docs:
                    vector_store.add_documents(documents=docs)
                else:
                    logger.warning("文档过滤后为空，跳过更新向量存储。")

        # 2. 初始化 LLM
        model = DeepSeekLLM(
            model="deepseek-r1:7b",
            base_url="http://192.168.64.1:11434"
        )

        # 3. 创建检索链（如果向量存储存在），否则构造默认的 prompt 链
        if vector_store:
            logger.info("创建检索链")
            retrieval_chain = initialize_retrieval_chain(vector_store, model)
            chat_runnable = retrieval_chain
        else:
            logger.info("检索链不存在，使用默认 prompt 构造对话链")
            retrieval_chain = initialize_simple_chain(model)
            chat_runnable = retrieval_chain
        return chat_runnable

    except Exception as e:
        logger.exception("准备聊天链时出错：%s", e)
        raise


def generate(
        file_path: Optional[str],
        user_input: str,
        session_id: str,
) -> Any:
    """
    主流程：非流式输出
    """
    try:
        chat_runnable = _prepare_chat_runnable(file_path, session_id)
        logger.info("生成非流式输出")
        if session_vector_stores.get(session_id) is None:
            return chat_with_history(chat_runnable, user_input, session_id, "base")
        return chat_with_history(chat_runnable, user_input, session_id, "retrieval")
    except Exception as e:
        logger.exception("生成非流式结果时出错：%s", e)
        raise


def generate_stream(
        file_path: Optional[str],
        user_input: str,
        session_id: str,
) -> Any:
    """
    主流程：流式输出
    """
    try:
        chat_runnable = _prepare_chat_runnable(file_path, session_id)
        logger.info("生成流式输出")
        if session_vector_stores.get(session_id) is None:
            yield from chat_with_history_stream(chat_runnable, user_input, session_id, "base")
        yield from chat_with_history_stream(chat_runnable, user_input, session_id, "retrieval")
    except Exception as e:
        logger.exception("生成流式结果时出错：%s", e)
        raise
