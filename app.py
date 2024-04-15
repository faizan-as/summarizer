import os
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv() 
st.set_page_config(page_title='RegIntel Summarizer')

logo_image_url = 'https://media.licdn.com/dms/image/C560BAQE7KIMIKXNi_Q/company-logo_200_200/0/1646756531216/tellence_logo?e=2147483647&v=beta&t=gmAcWjNBMuIbVfuv95MVRqkAhB56PFZIg9OE4_IX4fM'

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="{logo_image_url}" alt="Logo" style="width: 50px; height: 50px; margin-right: 10px;">
        <h1 style="margin: 0;">RegIntel Summarizer</h1>
    </div>
    """,
    unsafe_allow_html=True
)

def summarize(docs):
    llm = AzureChatOpenAI(temperature=0, deployment_name="gpt-35-turbo")
    # Define prompt
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    chain = load_summarize_chain(llm, chain_type="stuff")
    # docs= get_pdf_content(pdf_file_path)
    return stuff_chain.run(docs)

def get_pdf_content(uploaded_file):
    docs = []  # Create an empty list to store Document objects
    reader = PdfReader(uploaded_file)

    for page in reader.pages:
        text = page.extract_text()
        doc = Document(page_content=text)  # Create a Document object

        docs.append(doc)
    return docs

def main():
    #st.title("PDF Summarizer App")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
       with st.spinner("uploading..."):
        docs= get_pdf_content(uploaded_file)
        #st.success("file uploaded!")
       
    if st.button("Summarize PDF"):
        with st.spinner("Summarizing the uploaded PDF..."):
            summary = summarize(docs)
            st.write(summary)

if __name__ == "__main__":
    main()