import logging

from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_retrieval_chain(vector_store, llm, top_k: int = 5):
    """
    创建检索链，包含历史上下文的问答系统。

    :param vector_store: 向量存储对象
    :param llm: 使用的语言模型
    :param top_k: 检索的最大文档数，默认为5
    :return: 完整的检索链对象
    """
    if not vector_store:
        logger.error("向量存储为空，无法创建检索链。")
        raise ValueError("向量存储不能为空。")

    try:
        logger.info("初始化检索链...")

        # 配置文档检索器
        retriever = vector_store.as_retriever()

        # 1. 系统提示 - 检索上下文问答
        system_prompt = """
        你是一个用于问答任务的助手。  \
        使用以下检索到的上下文或对话历史来回答问题。\  
        如果你不知道答案，请直接说你不知道。  \
        回答必须使用markdown格式，并且保持回答简洁。\n\n{context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human", "{input}")
        ])

        # 创建文档链
        docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        logger.info("文档链初始化成功。")

        # 2. 上下文问题补全
        contextualize_q_prompt = """
        给定一个聊天历史和最新的用户问题，\
        该问题可能引用了聊天历史中的上下文，\  
        将其重新表述为一个独立的问题，  \
        使其在没有聊天历史的情况下也能被理解。\  
        不要回答这个问题，只需在需要时重新表述， \ 
        否则原样返回。
        """

        retriever_history_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human", "{input}")
        ])

        # 创建带历史上下文的检索器
        history_chain = create_history_aware_retriever(
            llm=llm,
            retriever=retriever.bind(k=top_k),
            prompt=retriever_history_prompt
        )
        logger.info("历史上下文检索器初始化成功，top_k=%d", top_k)

        # 创建最终的检索链
        retrieval_chain = create_retrieval_chain(retriever=history_chain, combine_docs_chain=docs_chain)
        logger.info("检索链创建成功。")

        return retrieval_chain

    except Exception as e:
        logger.error("初始化检索链失败: %s", e)
        raise RuntimeError(f"初始化检索链失败: {str(e)}")
