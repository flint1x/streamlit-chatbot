import streamlit as st
from typing import Generator, List, Dict


from openai import OpenAI
from typing import List, Dict

class DeepSeekClient:
    def __init__(self, api_key: str) -> None:
        """
        初始化大模型客户端。
        注意：base_url 必须指向 DeepSeek 的官方接口地址。
        """
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com"
        )

    def get_model_reply(self, messages: List[Dict[str, str]]) -> str:
        """
        将历史状态数组发送给服务器，并解析返回的结果。
        
        参数:
            messages: 包含了所有上下文的列表，例如 [{"role": "user", "content": "hi"}]
        返回:
            str: 模型生成的纯文本回复
        """
        try:
            # 发起网络请求
            response = self.client.chat.completions.create(
                model="deepseek-chat", # 指定使用的模型版本
                messages=messages,     # 直接传入我们维护的 session_state.messages
                max_tokens=2048,
                temperature=0.7        # 控制回答的发散程度
            )
            # 剥离复杂的 JSON 外壳，只提取核心的文本内容
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ API调用出现异常: {str(e)}"
    
    def get_stream_reply(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        流式请求大模型 API，返回一个生成器。
        每次 yield 产出一个字/词 (Token)。
        """
        # 1. 发起网络请求时，必须显式开启 stream=True 参数
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
        



class ChatbotConsole:
    def __init__(self) -> None:
        self.llm_client = DeepSeekClient(api_key=st.secrets["DEEPSEEK_API_KEY"])
        
        if "messages" not in st.session_state:
            # messages 列表用于存储所有的对话历史
            # 每个元素是一个字典，格式为 {"role": "角色", "content": "内容"}
            st.session_state.messages = []

    def render_history(self) -> None:
        """
        遍历并渲染历史对话记录。
        """
        for msg in st.session_state.messages:
            # st.chat_message("角色") 会自动生成对应的头像和气泡容器
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    def process_user_input(self) -> None:
        """
        监听用户输入，并生成模拟的大模型回复。
        """
        # st.chat_input 渲染底部输入框，如果有输入则返回字符串，否则返回 None
        user_text: str = st.chat_input("请在此输入您的问题...")
        
        if user_text:
            # 1. 保存并渲染用户的输入
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)
            
            # 2. 模拟大模型的响应逻辑（这里用固定字符串代替）
            with st.chat_message("assistant"):
                stream_generator = self.llm_client.get_stream_reply(st.session_state.messages)

                real_reply: str = st.write_stream(stream_generator)
            
            # 3. 保存并渲染机器人的回复
            st.session_state.messages.append({"role": "assistant", "content": real_reply})
    
    def render_sidebar(self) -> None:
        """
        渲染侧边栏组件并处理对应的交互逻辑。
        """
        st.sidebar.title("⚙️ 控制面板")
        
        # 思考题：如何把上面的 3 个知识点结合起来？
        # 1. 监听侧边栏按钮的点击事件
        if st.sidebar.button("🗑️ 清空记忆"):
            # 2. 清空状态列表中的历史记录
            st.session_state.messages = []
            # 3. 强制刷新页面，让 UI 立即呈现清空后的状态
            st.rerun()

    def run(self) -> None:
        """
        主控函数，定义页面组件的执行顺序。
        """
        st.title("💡 我的专属大模型控制台")
        self.render_sidebar()
        self.render_history()
        
        self.process_user_input()

if __name__ == "__main__":
    console = ChatbotConsole()
    console.run()