from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from chat_memory import get_chat_history
from Index_manager import IndexService
import settings as config


class QAService(object):
    def __init__(self):
        self.vector_service = IndexService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "你是一个严格基于知识库回答的助手。\n"
                 "你的回答必须完全基于下面提供的【检索到的参考资料】。\n"
                 "如果参考资料为空，或者参考资料中没有任何与用户问题相关的信息，\n"
                 "你必须直接回答：'知识库中暂未收录相关内容，我无法回答这个问题。'\n"
                 "禁止使用你自身的知识或常识来补充回答。\n"
                 "参考资料：\n{context}"
                 ),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        base_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_document(retriever.invoke(x["input"]))
            )
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            base_chain,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain


if __name__ == '__main__':
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = QAService().chain.invoke({"input": "我之前问了什么"}, session_config)
    print(res)