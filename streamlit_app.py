import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Streamlit 앱 설정
st.title("텍스트 요약기")
st.write("Langchain과 OpenAI를 사용한 텍스트 요약기입니다.")

# 사이드바에 API 키 입력창 추가
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# API 키가 제공되었는지 확인
if not api_key:
    st.warning("API Key를 입력해주세요.")
    st.stop()

# 요약을 위해 사용될 모델 초기화
llm = OpenAI(api_key=api_key)

# 텍스트 입력 받기
input_text = st.text_area("요약할 텍스트를 입력해주세요:")

# 요약 버튼
if st.button("요약하기"):
    if input_text:
        try:
            # Langchain 프롬프트 생성 및 요약 요청
            prompt = PromptTemplate(input_variables=["text"], template="요약해줘: {text}")
            summary = llm.invoke(prompt.format(text=input_text))
            
            # 요약 결과 출력
            st.subheader("요약 결과")
            st.write(summary)
        except Exception as e:
            st.error(f"요약 중 오류 발생: {e}")
    else:
        st.warning("요약할 텍스트를 입력해주세요.")