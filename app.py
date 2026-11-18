import os
import tempfile

import numpy as np
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")



st.set_page_config(
    page_title= "The VIP ChatBot",
    page_icon= "☠️",
    layout= "centered"
)

st.title("☠️ The VIP ChatBot")

st.write(
    "Upload a PDF and ask question about its content."
)


if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")

    st.stop()


llm = ChatGroq(
    api_key= GROQ_API_KEY,
    model= "openai/gpt-oss-20b",
    temperature=0
)

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name= "sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings

embeddings = load_embeddings()



uploaded_file = st.file_uploader(
    "Upload your PDF",
    type= ["pdf"]
)

# ------------------------------------------
# 9. PROCESS PDF
# -------------------------------------------
if uploaded_file:

    st.success("PDF upload successfully!")

    # ------------------------------------
    # Save uploaded PDF temmporarily
    # ------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        pdf_path = temp_file.name

    # -------------------------------------------
    # LOAD PDF
    # -------------------------------------------

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()
    
    # -------------------------------------------
    # SPLIT DOCUMENT INTO CHUNKS
    # -------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size= 500,
        chunk_overlap= 50
    )

    chunks = splitter.split_documents(documents)


    # --------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------

    text = [
        chunk.page_content
        for chunk in chunks
    ]

    document_embeddings = embeddings.embed_documents(
        text
    )

    document_embeddings = np.array(
        document_embeddings
    )


    st.success(
        f"PDF processed successfully! "
        f"{len(chunks)} chunkd createf. "
    )

    # ---------------------------------------------------
    # USER QUESTION
    # ---------------------------------------------------

    question = st.text_input(
        "Ask a question about your PDF:"
    )


    # ----------------------------------------------------
    # RAG PIPELINE
    # ----------------------------------------------------

    if question:

        # -----------------------------------------------
        # CREATE QUESTION EMBEDDING
        # ----------------------------------------------


        question_embedding = embeddings.embed_query(
            question
        )

        question_embedding = np.array(
            question_embedding
        )

        # -----------------------------------------------
        # CALCULATE COSINE SIMILARITY
        # -----------------------------------------------

        similarity_scores = (
            document_embeddings @ question_embedding
            ) / (
                np.linalg.norm(
                    document_embeddings,
                    axis= 1
                )
                *
                np.linalg.norm(
                    question_embedding
                )
        )


        # --------------------------------------------------
        # GET TOP 3 RELAVANT CHUNKS
        # --------------------------------------------------

        top_indices = np.argsort(
            similarity_scores
        )[-3:][::-1]


        relavant_chunks = [
            chunks[i]
            for i in top_indices
        ]


        # ---------------------------------------------------
        # CREATE CONTEXT
        # ---------------------------------------------------

        context = "\n\n".join(
            chunk.page_content
            for chunk in relavant_chunks
        )


        # ----------------------------------------------------
        # CREATE PROMPT
        # ----------------------------------------------------


        prompt = f"""
You are a helpful PDF assistant.

Anlswer the user's question using only
the information provided in the context.

If the answer is not present in the context,
say:

"I could not find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""


        # ----------------------------------------------------
        # SEND TO GROQ
        # ---------------------------------------------------

        with st.spinner("Generating answer..."):

            response = llm.invoke(prompt)


        # -------------------------------------------------
        # DISPLAY ANSWER
        # -------------------------------------------------

        st.subheader("Answer")

        st.write(response.content)