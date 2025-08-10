import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from zhipuai_llm import ZhipuaiLLM
from operator import itemgetter
from typing import List
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
api_key = 'your api key'
os.environ['ZHIPUAI_API_KEY'] = api_key
# 初始化模型和向量数据库（缓存避免重复初始化）
@st.cache_resource
def init_rag_chain():
    zhipuai_model = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1)
    embedding = ZhipuAIEmbeddings()
    persist_directory = 'path/to/myAgent/mydb'
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    # 初始化记忆库（实际部署建议使用Redis等持久化存储）
    session_store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    # 构建提示模板
    contextual_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是资深大模型系统工程师。请综合【知识库】与对话历史，给出准确、简洁、条理清晰的回答。\n【知识库】\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "当前问题：{input}")
    ])

    # 构建基础RAG链
    base_rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "history": itemgetter("history"),
        }
        | contextual_prompt
        | zhipuai_model
    )

    # 添加历史记忆
    conversational_chain = RunnableWithMessageHistory(
        base_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return conversational_chain

# 流式响应生成器
def generate_response(chain, input_text, session_id):
    config = {"configurable": {"session_id": session_id}}
    # 流式输出
    for chunk in chain.stream({"input": input_text}, config=config):
        yield chunk.content

# Streamlit应用界面
def main():
    # 1. 设置页面配置（应用名称修改）
    st.set_page_config(
        page_title="大模型知识问答助手",
        page_icon="🤖",
        layout="centered"
    )
    
    # 2. 添加右下角功能标注（使用CSS固定位置）
    st.markdown("""
    <style>
        .feature-badge {
            position: fixed;
            right: 20px;
            bottom: 20px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 8px 12px;
            border-radius: 12px;
            font-size: 0.8rem;
            z-index: 1000;
        }
    </style>
    <div class="feature-badge">
        ✅ 知识检索（大模型基础完整版.pdf）<br>
        ✅ 多轮对话<br>
        ✅ 流式输出
    </div>
    """, unsafe_allow_html=True)
    
    # 3. 主界面标题
    st.title("🤖 大模型知识问答助手")
    
    # 4. 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 5. 初始化对话链
    if "chain" not in st.session_state:
        with st.spinner("正在初始化知识库系统..."):
            st.session_state.chain = init_rag_chain()
    
    # 6. 固定session_id
    session_id = "user_123"
    
    # 7. 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 8. 用户输入处理
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 准备AI回复区域
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # 流式获取响应
            for chunk in generate_response(st.session_state.chain, prompt, session_id):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            # 最终显示（去掉光标）
            response_placeholder.markdown(full_response)
        
        # 添加AI回复到历史
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()