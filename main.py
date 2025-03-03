import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controller.file_controller import file_router
from controller.subscribe_controller import subscribe_router

# 创建主应用
app = FastAPI(
    title="文档摘要",
    description="这是一个关于文档摘要API的文档",
    version="1.0.0",
    contact={
        "name": "黄任翔",
        "mobile": "15236325327",
    }
)

# 将路由组包含到主应用中
app.include_router(subscribe_router)
app.include_router(file_router)

# 启用 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（你可以指定具体的来源列表）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, log_level='debug',
                reload=True)
