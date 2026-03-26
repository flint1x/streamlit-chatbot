import streamlit as st
from typing import Generator, List, Dict
import os
import tempfile

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

class DeepSeekClient:
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com"
        )
    
    def get_stream_reply(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        流式请求大模型 API，返回一个生成器。
        每次 yield 产出一个字/词 (Token)。
        """
        # 1. 发起网络请求时，必须显式开启 stream=True 参数
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True,          # 核心：开启流式传输
                max_tokens=2048,
                temperature=0.7
            )
            
            # 2. 此时 response 不再是一个包含完整结果的对象，而是一个可迭代的数据流
            for chunk in response:
                # 解析每个 chunk 中的增量文本 (delta.content)
                delta_content: str | None = chunk.choices[0].delta.content
                if delta_content is not None:
                    # 使用 yield 逐个吐出文字，而不是 return 一次性返回
                    yield delta_content
        except Exception as e:
            yield f"⚠️  API调用出现异常: {str(e)}"
        



class ChatbotConsole:
    def __init__(self) -> None:
        self.llm_client = DeepSeekClient(api_key=st.secrets["DEEPSEEK_API_KEY"])
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "processed_file_id" not in st.session_state:
                st.session_state.processed_file_id = None

        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        
    
    def render_sidebar(self) -> None:
        """
        渲染侧边栏组件
        """
        st.sidebar.title("⚙️ 控制面板")
        # [新增] PDF 上传组件
        st.sidebar.subheader("📚 知识库导入")
        uploaded_file = st.sidebar.file_uploader("上传 PDF 文档", type="pdf")
        
        # 如果有新文件上传，且还没被解析过
        if uploaded_file:
            file_id = uploaded_file.file_id

            if st.session_state.processed_file_id != file_id and st.session_state.vector_store is None:
                st.session_state.processed_file_id = file_id
                self._process_pdf(uploaded_file)
        
        if st.sidebar.button("🗑️ 清空记忆"):
            st.session_state.messages = []
            st.session_state.vector_store = None # 同时清空知识库
            st.rerun()

    def process_user_input(self) -> None:
        user_text: str = st.chat_input("你想聊些什么呢？")
        
        if user_text:
            # 将用户问题加入记忆历史
            st.session_state.messages.append({"role": "user", "content": user_text})

            # 渲染用户问题
            with st.chat_message("user"):
                st.markdown(user_text)
            
            

            # 寻找相关知识
            context_text = ""
            if st.session_state.vector_store:
                # 在知识库里寻找最相关的 3 个片段
                search_results = st.session_state.vector_store.similarity_search(user_text, k=3)
                context_text = "\n".join([doc.page_content for doc in search_results])

                if context_text:
                    with st.expander("查看检索到的背景知识"):
                        st.markdown(context_text)
                
            
            
            # 4. 调用大模型（注意：我们只在发送时增强 Prompt，存进 history 的还是原问题）
            with st.chat_message("assistant"):
                if context_text:
                    system_content = (
                        "你是一位专业助手，擅长根据提供的文档资料回答问题。\n"
                        "回答要求：\n"
                        "1. 优先基于以下背景知识作答\n"
                        "2. 如背景知识不足，可结合自身知识补充，但需注明'以下为补充信息'\n"
                        "3. 回答简洁、准确，避免无关内容\n\n"
                        f"背景知识：\n{context_text}"
                    )
                else:
                    system_content = "你是一位专业、友好的助手。回答要简洁准确，遇到不确定的问题请如实说明。"
                temp_messages = [{"role": "system", "content": system_content}]
                temp_messages.extend(st.session_state.messages[:-1]) # 拿走刚才存的 user_text
                temp_messages.append({"role": "user", "content": user_text})
                stream_generator = self.llm_client.get_stream_reply(temp_messages)
                real_reply: str = st.write_stream(stream_generator)

            st.session_state.messages.append({"role": "assistant", "content": real_reply})

    def render_history(self) -> None:
        """
        遍历并渲染历史对话记录。
        """
        for msg in st.session_state.messages:
            # st.chat_message("角色") 会自动生成对应的头像和气泡容器
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])



    def _process_pdf(self, uploaded_file) -> None:
        """
        [新增] 核心函数：解析 PDF 并存入向量数据库
        """
        with st.spinner("🚀 正在深度解析 PDF 并构建索引..."):
            # A. 将上传的文件写入临时物理路径（LangChain 的 Loader 需要路径）
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # B. 加载 PDF 文本
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # C. 文本切片（防止单段太长导致模型记不住）
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # D. 向量化并存入 FAISS (使用免费的 HuggingFace 模型)
            # 注意：第一次运行会自动下载模型（约100MB），之后就快了
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
                                          api_key=st.secrets["OPENAI_API_KEY"],
                                          base_url=st.secrets["OPENAI_BASE_URL"] )
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            
            # E. 清理临时文件
            os.remove(tmp_path)
            st.success("✅ 知识库已就绪！现在可以针对文档提问了。")

    


    def run(self) -> None:
        """
        主控函数，定义页面组件的执行顺序。
        """
        st.title("🦙chatbot123123131")
        self.render_sidebar()
        self.render_history()
        
        self.process_user_input()

if __name__ == "__main__":
    console = ChatbotConsole()
    console.run()