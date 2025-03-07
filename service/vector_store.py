import logging
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from service.document_processor import load_and_split_document
from service.sql import init_conversation_messages

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 获取当前脚本的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前路径的父路径
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)

# 🛠 配置部分
persist_directory = f"{grand_path}/chroma_db"  # Chroma 本地持久化目录
embeddings = HuggingFaceEmbeddings(
    model_name=f'{grand_path}/models/bge-large-zh-v1.5',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 🟢 初始化本地向量数据库（如果已存在则加载）
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

init_conversation_messages()


# 📁 计算上传顺序的函数 (自动计算 upload_order)
def get_next_upload_order(user_id, session_id):
    # 首先获取符合条件的总记录数
    results = vector_store.get(where={
        "$and": [
            {"user_id": user_id, },
            {"session_id": session_id, }
        ]
    })
    metadatas = results["metadatas"] if results else None
    last_record = metadatas[-1] if metadatas else None
    next_order = (last_record.get("upload_order", 0) if last_record else 0) + 1
    return next_order


# 📁 上传文档的函数 (自动获取 upload_order)
def upload_file(file_path, user_id, session_id):
    # 自动获取当前用户和会话的下一个上传顺序
    upload_order = get_next_upload_order(user_id, session_id)

    # 加载和分割文档
    split_docs = load_and_split_document(file_path)

    # 添加元数据 (自动计算的 upload_order)
    for doc in split_docs:
        doc.metadata["user_id"] = user_id
        doc.metadata["session_id"] = session_id
        doc.metadata["file_path"] = file_path
        doc.metadata["upload_order"] = upload_order

    # 添加文档到向量数据库并持久化
    vector_store.add_documents(split_docs)
    print(f"📥 文件 '{file_path}' 已上传并存储到数据库！ (upload_order={upload_order})")
