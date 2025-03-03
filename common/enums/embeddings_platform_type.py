from enum import Enum


class EmbeddingsPlatformType(Enum):
    # 平台枚举定义，包含平台名称和对应处理实体
    HUGGING_FACE = "huggingface"

    def __init__(self, code: str):
        # 初始化平台枚举值
        self.code = code

    def code(self) -> str:
        # 获取平台的显示名称
        return self.code
