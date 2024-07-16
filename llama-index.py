import os
import streamlit as st
from dotenv import load_dotenv
import openai

from llama_index.core import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

#load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets.Secrets.key

def main():
    st.header("PDFの内容に関する質問をしてください")
    
    input_dir = "./data"
    
    try:
        reader = SimpleDirectoryReader(input_dir=input_dir,recursive=True)
        docs = reader.load_data()
        
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo",temperature=0.5,
                       system_prompt="あなたは提供されたデータの専門家です。ユーザーの質問に対して、的確な答えを出してください")
        )
        index = VectorStoreIndex.from_documents(docs, service_context = service_context)
        
        query = st.text_input("pdfの内容に関する質問を入力してください")
        
        if query:
            chat_engine = index.as_chat_engine(chat_mode="condense_question",verbose=True)
            response = chat_engine.chat(query)
            
            st.markdown("**回答**")
            st.write(response.response.strip())
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        
if __name__ == "__main__":
    main()
            