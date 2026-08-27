import streamlit as st
from qa_engine import QAService
import settings as config
from dotenv import load_dotenv
load_dotenv()

st.title("药康助手")
st.divider()

# 初始化 session_state
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，我是药康助手，擅长回答用药安全、慢病管理和健康知识相关的问题，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = QAService()

# 显示历史消息
for message in st.session_state["message"]:
    if message["role"] == "user":
        avatar = "🧑‍⚕️"   # 用户头像
    else:
        avatar = "💊"     # 助手头像
    st.chat_message(message["role"], avatar=avatar).write(message["content"])
prompt = st.chat_input()

if prompt:
    # 显示用户消息
    st.chat_message("user", avatar = "🧑‍⚕️").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 检索，获取引用文档
    try:
        # 通过 vector_service 获取检索器（混合检索器）
        retriever = st.session_state["rag"].vector_service.get_retriever()
        retrieved_docs = retriever.invoke(prompt)  # 或者 .get_relevant_documents(prompt)
        st.session_state["last_docs"] = retrieved_docs
    except Exception as e:
        st.warning(f"获取引用文档失败: {e}")
        retrieved_docs = []

    ai_res_list = []
    with st.spinner("AI 思考中......."):
        res_stream = st.session_state["rag"].chain.stream(
            {"input": prompt},
            config.session_config
        )

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        # 流式 AI 回答
        st.chat_message("assistant", avatar="💊").write_stream(capture(res_stream, ai_res_list))
        full_answer = "".join(ai_res_list)
        st.session_state["message"].append({"role": "assistant", "content": full_answer})

    # 引用原文折叠
    if "知识库中暂未收录" not in full_answer and retrieved_docs :
        with st.expander("📄 查看引用的参考资料原文", expanded=False):
            for i, doc in enumerate(retrieved_docs, 1):
                # 如果有元数据（如来源、页码）可一并显示
                source = doc.metadata.get("source", f"文档{i}")
                st.markdown(f"**引用 {i}：{source}**")
                st.write(doc.page_content)
                st.divider()