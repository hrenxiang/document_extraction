import logging
import os

from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader, PyPDFLoader,
)
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from rapidocr_onnxruntime import RapidOCR

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 负责文档加载和拆分
def load_and_split_document(file_path: str):
    """
    根据文件类型加载文档并拆分成多个 Document 对象
    """
    try:
        ext = os.path.splitext(file_path)[-1].lower()
        logger.info("加载文件：%s, 扩展名：%s", file_path, ext)

        if ext == '.txt':
            return load_txt_splitter(file_path)
        elif ext == '.md':
            return load_md_splitter(file_path)
        elif ext in ['.doc', '.docx']:
            return load_word_splitter(file_path)
        elif ext == '.pdf':
            return load_pdf_splitter(file_path)
        elif ext == '.jpg':
            return load_jpg_splitter(file_path)
        else:
            logger.error("不支持的文件类型：%s", ext)
            return []
    except Exception as e:
        logger.exception("加载并拆分文档时出错：%s", e)
        return []


# 加载 TXT 文件
def load_txt_file(file_path: str) -> list:
    try:
        logger.info("加载TXT文件：%s", file_path)
        loader = UnstructuredLoader(file_path)
        docs = loader.load()
        return docs
    except Exception as e:
        logger.exception("加载TXT文件时出错：%s", e)
        return []


# 加载 Markdown 文件
def load_md_file(file_path: str) -> list:
    try:
        logger.info("加载Markdown文件：%s", file_path)
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        return docs
    except Exception as e:
        logger.exception("加载Markdown文件时出错：%s", e)
        return []


# 加载 Word 文件
def load_word_file(file_path: str) -> list:
    try:
        logger.info("加载Word文件：%s", file_path)
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        docs = loader.load()
        return docs
    except Exception as e:
        logger.exception("加载Word文件时出错：%s", e)
        return []


# 加载 PDF 文件
def load_pdf_file(file_path: str) -> list:
    try:
        logger.info("加载PDF文件：%s", file_path)
        # 设置 OCR 相关环境变量
        os.environ['OCR_AGENT'] = 'unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract'
        loader = PyPDFLoader(file_path=file_path)
        docs = loader.load()
        return docs
    except Exception as e:
        logger.exception("加载PDF文件时出错：%s", e)
        return []


# 加载 JPG 文件（通过 OCR 识别）
def load_jpg_file(file_path: str) -> str:
    try:
        logger.info("加载JPG文件：%s", file_path)
        ocr = RapidOCR()
        result, _ = ocr(file_path)
        if result:
            ocr_result = [line[1] for line in result]
            return "\n".join(ocr_result)
        else:
            logger.warning("JPG文件OCR未返回结果：%s", file_path)
            return ""
    except Exception as e:
        logger.exception("加载JPG文件时出错：%s", e)
        return ""


# 预处理文本（例如：转换小写、去除标点等）
def preprocess_text(text: str) -> str:
    try:
        logger.info("预处理文本")
        text = text.lower()
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        return text
    except Exception as e:
        logger.exception("预处理文本时出错：%s", e)
        return text


# 分割 TXT 文件
def load_txt_splitter(txt_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分TXT文件：%s", txt_file)
        docs = load_txt_file(txt_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.exception("拆分TXT文件时出错：%s", e)
        return []


# 分割 Markdown 文件
def load_md_splitter(md_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分Markdown文件：%s", md_file)
        docs = load_md_file(md_file)
        text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.exception("拆分Markdown文件时出错：%s", e)
        return []


# 分割 Word 文件
def load_word_splitter(word_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分Word文件：%s", word_file)
        docs = load_word_file(word_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.exception("拆分Word文件时出错：%s", e)
        return []


# 分割 PDF 文件
def load_pdf_splitter(pdf_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分PDF文件：%s", pdf_file)
        docs = load_pdf_file(pdf_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.exception("拆分PDF文件时出错：%s", e)
        return []


# 分割 JPG 文件（OCR后处理）
def load_jpg_splitter(jpg_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分JPG文件：%s", jpg_file)
        docs = load_jpg_file(jpg_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # 因为 OCR 返回的是字符串，所以使用 create_documents 将文本转换成 Document 对象
        split_docs = text_splitter.create_documents([docs])
        return split_docs
    except Exception as e:
        logger.exception("拆分JPG文件时出错：%s", e)
        return []
