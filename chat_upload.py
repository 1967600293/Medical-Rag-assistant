import streamlit as st
from db_service import HealthDataService
from dotenv import load_dotenv
load_dotenv()

st.title("📚 知识库更新")
#初始化知识服务
if "service" not in st.session_state:
    st.session_state["service"] = HealthDataService()
#上传多个文件
uploaded_files = st.file_uploader(
    "上传 TXT 文件（可多选）",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    total = len(uploaded_files)
    progress = st.progress(0)
    status = st.empty()

    success = 0
    failed = []

    for i, f in enumerate(uploaded_files):
        progress.progress((i + 1) / total)
        status.info(f"处理中：{f.name}")

        try:
            content = f.getvalue().decode("utf-8")
            result = st.session_state["service"].upload_by_str(content, f.name)
            status.success(f"✅ {f.name} 成功")
            success += 1
        except UnicodeDecodeError:
            status.error(f"❌ {f.name} 编码错误（需 UTF-8）")
            failed.append(f.name)
        except Exception as e:
            status.error(f"❌ {f.name} 失败：{e}")
            failed.append(f.name)
    # 清空状态占位，显示最终汇总
    status.empty()
    if failed:
        st.warning(f"成功 {success} 个，失败 {len(failed)} 个：{', '.join(failed)}")
    else:
        st.success(f"全部 {total} 个文件上传成功")