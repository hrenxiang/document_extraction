import logging
from typing import Any, Generator

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# 配置日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 全局会话历史存储字典
store = {}  # 存储历史对话


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取会话历史记录，如果不存在则初始化一个新的 ChatMessageHistory。
    """
    if session_id not in store:
        logger.info("会话 %s 不存在历史记录，初始化新记录。", session_id)
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def clean_session_history(session_id: str):
    store.pop(session_id, None)


def _get_result_chain(retrieval_chain: Any) -> RunnableWithMessageHistory:
    """
    根据检索链和会话ID构建带历史记录管理的 Runnable 对象。
    """
    return RunnableWithMessageHistory(
        runnable=retrieval_chain,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    )


def chat_with_history_stream(
        retrieval_chain: Any,
        user_input: str,
        session_id: str,
        type: str,
) -> Generator[Any, None, None]:
    """
    进行带历史记录的对话，支持流式输出。

    :param retrieval_chain: 检索链或对话链对象
    :param user_input: 用户输入
    :param session_id: 会话ID
    :param type: 检索链类型
    :yield: 逐步生成的对话输出
    """
    config = {'configurable': {'session_id': session_id}}
    try:
        result_chain = _get_result_chain(retrieval_chain)
        logger.info("开始流式对话处理，session_id=%s", session_id)
        for item in result_chain.stream(input={'input': user_input}, config=config):
            yield item
    except Exception as e:
        logger.exception("调用 chat_with_history_stream 时发生异常：%s", e)
        raise
