import streamlit as st
import os
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

# 1. 페이지 설정 및 보안
st.set_page_config(page_title="금융 보고서 RAG 챗봇", page_icon="💰")
st.title("💰 금융 보고서 RAG 챗봇")

# OpenAI API 키 설정 (환경 변수 또는 사이드바 입력)
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key # 입력 즉시 환경변수 설정

# 2. 초기화 로직 (한 번만 실행되도록 캐싱)
@st.cache_resource
def prepare_rag_system(file_path):
    # [1단계] 문서 로딩 및 구조 기반 청킹 [cite: 939-953]
    converter = DocumentConverter()
    result = converter.convert(source=file_path)
    doc = result.document
    
    hc = HierarchicalChunker()
    chunks = list(hc.chunk(dl_doc=doc))
    
    # [2단계] 임베딩 및 벡터 DB 구축 [cite: 957-970]
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_texts(
        [c.text for c in chunks],
        embedding=embedding,
        collection_name="finance_docs"
    )
    
    # [3단계] 하이브리드 검색기 구성 [cite: 981-993]
    bm25_retriever = BM25Retriever.from_texts([c.text for c in chunks])
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    return ensemble_retriever

# 3. 메인 인터페이스
uploaded_file = st.file_uploader("분석할 금융 PDF 파일을 업로드하세요", type="pdf")

if uploaded_file and os.environ.get("OPENAI_API_KEY"):
    # 파일 임시 저장
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("보고서를 분석하고 벡터 DB를 구축 중입니다..."):
        retriever = prepare_rag_system("temp.pdf")
    
    # [4단계] 메모리 및 대화 체인 설정 [cite: 685-710]
    if "memory" not in st.session_state:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        st.session_state.memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
            retriever=retriever,
            memory=st.session_state.memory,
            return_source_documents=True
        )

    # 5. 채팅 UI 구현
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                # RAG 파이프라인 실행 [cite: 1006]
                response = st.session_state.qa_chain.invoke({"question": prompt})
                answer = response['answer']
                st.markdown(answer)
                
                # 출처 표시 확장 아이디어 적용 
                with st.expander("참고 문서 확인"):
                    for i, doc in enumerate(response['source_documents']):
                        st.write(f"**근거 {i+1}**: {doc.page_content[:200]}...")
            
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("시작하려면 OpenAI API 키를 입력하고 PDF 파일을 업로드해 주세요.")