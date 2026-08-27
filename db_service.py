"""
知识库管理模块
"""
import os
import hashlib
from datetime import datetime
import settings as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_md5(text: str) -> str:
    """计算字符串的 MD5"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_processed(md5: str) -> bool:
    """检查 MD5 是否已记录"""
    if not os.path.exists(config.md5_path):
        return False
    with open(config.md5_path, "r", encoding="utf-8") as f:
        return md5 in set(line.strip() for line in f)


def mark_processed(md5: str) -> None:
    """记录已处理的 MD5"""
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5 + "\n")


class HealthDataService:
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)

        self.embedding = DashScopeEmbeddings(model="text-embedding-v4")
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self, content: str, filename: str) -> str:
        """上传文本到知识库，返回操作结果"""
        md5 = get_md5(content)
        print(f"[DEBUG] MD5: {md5}, 是否已处理: {is_processed(md5)}")
        if is_processed(md5):
            return "[Repeat] 内容已存在"

        # 分块
        if len(content) > config.max_spliter_char_number:
            chunks = self.splitter.split_text(content)
        else:
            chunks = [content]

        # 元数据
        meta = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "客户",
        }

        # 入库
        self.chroma.add_texts(chunks, metadata=[meta.copy() for _ in chunks])
        mark_processed(md5)
        return "[Success] 已载入向量库"


if __name__ == "__main__":
    svc = HealthDataService()
    print(svc.upload_by_str("项目", "testfile"))