# import os
# from dotenv import load_dotenv
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.embeddings import HuggingFaceEmbeddings

# def main():
#     load_dotenv()
    
#     if not os.getenv("GOOGLE_API_KEY"):
#         st.error("Google API Key missing from .env file!")
#         st.stop()

#     st.set_page_config(page_title="Bilingual PDF RAG Bot", layout="centered")
#     st.header("Ask your PDF 💬")
    
#     pdf = st.file_uploader("Upload your PDF file", type="pdf")
    
#     if pdf is not None:
#         db_folder = f"faiss_db_{os.path.splitext(pdf.name)[0]}"
        
#         with st.spinner("Loading embedding model..."):
#             embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#             )

#         if os.path.exists(db_folder):
#             with st.spinner("Found cached vectors! Loading database..."):
#                 knowledge_base = FAISS.load_local(
#                     db_folder, 
#                     embeddings, 
#                     allow_dangerous_deserialization=True
#                 )
#             st.success("Document loaded instantly from Cache!")
        
#         else:
#             with st.spinner("First time setup: Extracting text from PDF..."):
#                 pdf_reader = PdfReader(pdf)
#                 text = ""
#                 for page in pdf_reader.pages:
#                     text += page.extract_text()
                
#             text_splitter = RecursiveCharacterTextSplitter(
#                 separators=["\n\n", "\n", " ", ""],
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 length_function=len
#             )
#             chunks = text_splitter.split_text(text)
            
#             with st.spinner("First time setup: Generating embeddings locally..."):
#                 knowledge_base = FAISS.from_texts(chunks, embeddings)
#                 knowledge_base.save_local(db_folder)
                
#             st.success("Document processed and cached successfully!")
        
#         user_question = st.text_input("Ask a question about your PDF:")
        
#         if user_question:
#             with st.spinner("Generating answer via Gemini..."):
#                 docs = knowledge_base.similarity_search(user_question, k=3)
                
#                 llm = ChatGoogleGenerativeAI(
#                     model="gemini-1.5-flash", 
#                     temperature=0.3
#                 )
#                 chain = load_qa_chain(llm, chain_type="stuff")
#                 response = chain.run(input_documents=docs, question=user_question)
                   
#                 st.write("### 🤖 Response:")
#                 st.write(response)

# if __name__ == '__main__':
#     main()
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

# Modern LangChain 1.0+ ecosystem imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main():
    # Page configuration MUST be the very first Streamlit command executed
    st.set_page_config(page_title="Bilingual PDF RAG Bot", layout="centered")
    
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API Key missing from .env file!")
        st.stop()

    st.header("Ask your PDF 💬")
    
    pdf = st.file_uploader("Upload your PDF file", type="pdf")
    
    if pdf is not None:
        db_folder = f"faiss_db_{os.path.splitext(pdf.name)[0]}"
        
        with st.spinner("Loading embedding model..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

        if os.path.exists(db_folder):
            with st.spinner("Found cached vectors! Loading database..."):
                knowledge_base = FAISS.load_local(
                    db_folder, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
            st.success("Document loaded instantly from Cache!")
        
        else:
            with st.spinner("First time setup: Extracting text from PDF..."):
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    if page.extract_text():
                        text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            with st.spinner("First time setup: Generating embeddings locally..."):
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                knowledge_base.save_local(db_folder)
                
            st.success("Document processed and cached successfully!")
        
        user_question = st.text_input("Ask a question about your PDF:")
        
        if user_question:
            with st.spinner("Generating answer via Gemini..."):
                # Retrieve the top 3 relevant chunks
                docs = knowledge_base.similarity_search(user_question, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                
                # Setup modern Chat Model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.5-flash", 
                    temperature=0.3
                )
                
                # Explicit Prompt Template for RAG
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer the user's question using only the context provided below. If you do not know the answer based on the context, say that you cannot find it in the document.\n\nContext:\n{context}"),
                    ("human", "{question}")
                ])
                
                # Native Modern LCEL Pipeline
                chain = prompt | llm | StrOutputParser()
                
                # Execute the pipeline
                response = chain.invoke({"context": context, "question": user_question})
                    
                st.write("### 🤖 Response:")
                st.write(response)

if __name__ == '__main__':
    main()