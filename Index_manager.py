from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import settings as config
import streamlit as st
import jieba
from log_config import default_logger as logger

class IndexService(object):
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
        # 先不急着加载 BM25，等真正调用 get_retriever 时再搞
        self._bm25_retriever = None

    @staticmethod
    def _chinese_tokenizer(text):
        """BM25 中文分词适配（不加这个，BM25在中文下就是个摆设）"""
        return list(jieba.cut_for_search(text))  # 搜索引擎模式切词

    def _get_or_build_bm25(self):
        """懒加载 BM25，第一次调用时才会去拿数据（避免启动卡死）"""
        if self._bm25_retriever is not None:
            return self._bm25_retriever

        # 用 streamlit 的缓存把加载数据的动作缓存起来，第二次启动直接读缓存
        @st.cache_resource(show_spinner=False)
        def _load_bm25_from_db():
            all_docs = self.vector_store.get(include=["documents", "metadatas"])
            if not all_docs["documents"]:
                logger.info("Chroma 无文档，BM25 未构建")
                return None
            doc_objects = [
                Document(
                    page_content=doc,
                    metadata=meta if meta is not None else {}  # 关键修复
                )
                for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
            ]
            logger.info(f"BM25 加载完成，文档片段数: {len(doc_objects)}")
            return BM25Retriever.from_documents(
                doc_objects,
                k=config.similarity_threshold,
                preprocess_func=self._chinese_tokenizer
            )

        self._bm25_retriever = _load_bm25_from_db()
        return self._bm25_retriever

    def get_retriever(self):
        # 向量检索器（保持原样）
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.similarity_threshold}
        )

        bm25 = self._get_or_build_bm25()
        if not bm25:
            # 现实场景：没数据就直接报个错或者只用向量，没必要搞花里胡哨的降级
            return vector_retriever

        # 混合权重：语义检索权重稍高一点，关键词辅助
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25],
            weights=[0.6, 0.4]  # 经过测试微调出来的经验值
        )