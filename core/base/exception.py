# 统一响应模型
from typing import Optional, Any

from pydantic import BaseModel


class ResponseModel(BaseModel):
    code: int = 200  # 状态码，200 表示成功
    message: str = "成功"  # 提示信息
    data: Optional[Any] = None  # 具体数据
