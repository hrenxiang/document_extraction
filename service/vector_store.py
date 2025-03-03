import logging
import os
import traceback

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 获取当前脚本的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前路径的父路径
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)

def initialize_vector_store(text_chunks):
    """
    创建并返回向量存储。

    :param text_chunks: 待处理的文本块列表
    :return: 初始化完成的向量存储对象
    """
    model_path = f'{grand_path}/models/bge-large-zh-v1.5'

    try:
        logger.info("初始化向量存储，模型路径：%s", model_path)

        # 初始化 Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 过滤和转换文档
        docs = filter_complex_metadata(text_chunks)
        if not docs:
            logger.warning("文档过滤后为空，无法创建向量存储。")
            return None

        # 创建向量存储
        vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
        logger.info("向量存储初始化成功，包含 %d 个文档。", len(docs))

        return vector_store

    except Exception as e:
        error_message = traceback.format_exc()
        logger.error("初始化向量存储失败: %s", e)
        logger.debug("详细错误信息: %s", error_message)
        raise RuntimeError(f"初始化向量存储失败: {str(e)}")
