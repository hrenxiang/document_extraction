import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from service.chat_history import get_session_history
from service.chat_service import generate_stream
from service.sql import insert_into_conversation_messages, query_session_history, query_next_qa_id

# 创建一个路由组
subscribe_router = APIRouter(prefix="/subscribe")


@subscribe_router.get("/", summary="流式响应接口")
def subscribe(user_input: str, user_id: str, session_id: str, file_path: Optional[str] = None) -> StreamingResponse:
    unique_id = str(uuid.uuid4())
    insert_into_conversation_messages(
        user_id=user_id,
        session_id=session_id,
        parent_id=0,
        message_id=unique_id,
        qa_id=query_next_qa_id(user_id=user_id, session_id=session_id),
        qa_type="question",
        message_content=user_input
    )

    ret = generate_stream(file_path=file_path, user_input=user_input, user_id=user_id, session_id=session_id)
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
        result = ""
        for token in ret:
            # 统一处理 token，无论是字符串还是字典形式
            if isinstance(token, dict) and "answer" in token:
                result += token['answer']
                yield format_message(token['answer'], finished=False)
            elif isinstance(token, str):
                result += token
                yield format_message(token, finished=False)

        # 发送结束信号
        insert_into_conversation_messages(
            user_id=user_id,
            session_id=session_id,
            parent_id=0,
            message_id=unique_id,
            qa_id=query_next_qa_id(user_id=user_id, session_id=session_id),
            qa_type="answer",
            message_content=result
        )
        yield format_message("[DONE]", finished=True)

    return StreamingResponse(predict(), media_type="text/event-stream")


@subscribe_router.get("/create_session")
def create_session(user_input: str, user_id: str, ):
    hex_ = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    unique_id = str(uuid.uuid4())
    insert_into_conversation_messages(
        user_id=user_id,
        session_id=str(hex_),
        parent_id=0,
        message_id=unique_id,
        qa_id=0,
        qa_type="answer",
        message_content=user_input
    )


@subscribe_router.get("/search_history")
def session_history(user_id: str, session_id: str):
    return query_session_history(user_id, session_id)
