import logging
import traceback
from typing import Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings

from common.enums.embeddings_platform_type import EmbeddingsPlatformType

# 配置日志记录
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_embeddings(model_path: str, platform_type: str, model_kwargs: Dict[str, Any], encode_kwargs: Dict[str, Any], ):
    """
    获取嵌入模型实例。

    Args:
        model_path (str): 模型路径或名称。
        platform_type (str): 平台类型 (如 'huggingface')。
        model_kwargs (Dict[str, Any]): 模型初始化参数。
        encode_kwargs (Dict[str, Any]): 编码参数。

    Returns:
        object: 对应的嵌入模型实例。

    Raises:
        ValueError: 如果 platform_type 不合法。
    """

    # 检查平台类型，返回不同Embedding实例
    try:
        if EmbeddingsPlatformType.HUGGING_FACE.code() == platform_type:
            return HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    except Exception as e:
        error_message = traceback.format_exc()
        logger.error("向量模型初始化失败: %s", e)
        logger.debug("向量模型初始化失败详细错误信息: %s", error_message)
        raise ValueError("向量模型初始化失败")
    raise ValueError("暂不支持此平台提供的Embedding")
