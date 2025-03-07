import os
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from starlette.responses import FileResponse

from core.base.exception import ResponseModel
from service.vector_store import upload_file

# 创建一个路由组
file_router = APIRouter(prefix="/file")


# 上传文件到服务器本地
@file_router.post("/upload/", summary="上传文件接口")
async def upload(file: UploadFile = File(...),
                 user_id: str = Form(...),
                 session_id: str = Form(...)):
    content = await file.read()  # 读取整个文件内容
    max_size = 10 * 1024 * 1024  # 例如：2KB
    if len(content) > max_size:
        return ResponseModel(message="文件过大，最大10MB", code=500)

    # 目标文件路径
    upload_path = f"./uploads/{session_id}/{file.filename}"

    # 创建上传目录，如果不存在的话
    os.makedirs(os.path.dirname(upload_path), exist_ok=True)

    # 重置文件指针到开头，防止已经被读取过
    file.file.seek(0)

    # 使用 'w' 模式打开文件，会清空原文件内容
    with open(upload_path, 'w', encoding='utf-8') as fp:
        print('hello world', file=fp)

    upload_file(file_path=upload_path, user_id=user_id, session_id=session_id)

    return ResponseModel(message=f"File {file.filename} uploaded successfully!", data=upload_path)


# 下载文件
@file_router.get("/download", summary="下载文件接口")
async def download_file(session_id: str, filename: str):
    file_path = f"./uploads/{session_id}/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)
