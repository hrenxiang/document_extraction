import json
import os
import shutil
from typing import Optional

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from core.base.exception import ResponseModel
from service.chat_history import get_session_history, clean_session_history
from service.chat_service import generate, generate_stream, clean_session_vector_stores

# 创建一个路由组
subscribe_router = APIRouter(prefix="/subscribe")


@subscribe_router.get("/", summary="流式响应接口")
def subscribe(user_input: str, session_id: str, file_path: Optional[str] = None) -> StreamingResponse:
    ret = generate_stream(file_path=file_path, user_input=user_input, session_id=session_id)
    session_history = get_session_history(session_id)
    qaId = len(session_history.messages) // 2 + 1

    def format_message(data: str, finished: bool) -> str:
        """格式化流式输出的消息数据"""
        js_data = {
            "sceneName": "文档提取",
            "finished": "true" if finished else "false",
            "data": data,
            "answerRenderType": "markdown",
            "qaId": qaId
        }
        return f"data: {json.dumps(js_data, ensure_ascii=False)}\n\n"

    def predict():
        """流式返回生成的内容"""

        for token in ret:
            # 统一处理 token，无论是字符串还是字典形式
            if isinstance(token, dict) and "answer" in token:
                yield format_message(token['answer'], finished=False)
            elif isinstance(token, str):
                yield format_message(token, finished=False)

        # 发送结束信号
        yield format_message("[DONE]", finished=True)

    return StreamingResponse(predict(), media_type="text/event-stream")


@subscribe_router.get(path="/sync", summary="同步响应接口")
def subscribe_sync(user_input: str, session_id: str, file_path: Optional[str] = None):
    ret = generate(file_path=file_path, user_input=user_input, session_id=session_id)
    return ResponseModel(data=ret)


@subscribe_router.get(path="/clean", summary="清除缓存接口")
def clean(session_id: str):
    clean_session_vector_stores(session_id)
    clean_session_history(session_id)

    # 目标文件路径
    upload_path = f"./uploads/{session_id}"
    # 检查目录是否存在并确保是目录
    if os.path.exists(upload_path) and os.path.isdir(upload_path):
        try:
            # 递归删除目录及其所有内容
            shutil.rmtree(upload_path)
            print(f"成功删除目录: {upload_path}")
        except Exception as e:
            print(f"删除 {upload_path} 时发生错误: {e}")
    else:
        print(f"目录 {upload_path} 不存在或不是一个有效的目录")
