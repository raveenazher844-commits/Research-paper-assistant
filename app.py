
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# Load API key from the .env file
load_dotenv()

# Configure the Streamlit page
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# Main heading
st.title("📚 Research Paper Assistant")
st.write(
    "Upload a research paper PDF and ask questions "
    "about its contents using Retrieval-Augmented Generation (RAG)."
)


# Create embeddings model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Read PDF and create the vector database
def process_pdf(uploaded_file):
    os.makedirs("data", exist_ok=True)

    pdf_path = os.path.join("data", uploaded_file.name)

    with open(pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No readable pages were found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No readable text found. The PDF may be scanned."
        )

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, len(documents), len(chunks)


# Upload section
uploaded_file = st.file_uploader(
    "Upload your research paper",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Process Research Paper"):
        try:
            with st.spinner("Reading PDF and creating embeddings..."):
                vectorstore, page_count, chunk_count = process_pdf(
                    uploaded_file
                )

                st.session_state["vectorstore"] = vectorstore
                st.session_state["pdf_name"] = uploaded_file.name

            st.success("Research paper processed successfully!")

            st.write(f"Pages processed: {page_count}")
            st.write(f"Text chunks created: {chunk_count}")

        except Exception as error:
            st.error(f"Could not process PDF: {error}")


# Question and answer section
if "vectorstore" in st.session_state:

    st.subheader("Ask Questions About Your Paper")

    question = st.text_input(
        "Enter your research question:",
        placeholder="What is the main objective of this paper?"
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question first.")

        else:
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                st.error(
                    "GROQ_API_KEY is missing. "
                    "Add your API key to the .env file."
                )

            else:
                try:
                    with st.spinner("Finding relevant information..."):

                        # Retrieve the most relevant text chunks
                        retriever = st.session_state[
                            "vectorstore"
                        ].as_retriever(
                            search_kwargs={"k": 4}
                        )

                        relevant_docs = retriever.invoke(question)

                        if not relevant_docs:
                            st.warning(
                                "No relevant information was found."
                            )

                        else:
                            # Combine retrieved text into context
                            context = "\n\n".join(
                                [
                                    f"Page "
                                    f"{doc.metadata.get('page', 0) + 1}:\n"
                                    f"{doc.page_content}"
                                    for doc in relevant_docs
                                ]
                            )

                            # Connect to the language model
                            llm = ChatGroq(
                                model="llama-3.3-70b-versatile",
                                temperature=0
                            )

                            # Build the RAG prompt
                            prompt = f"""
You are a research paper assistant.

Answer the question using ONLY the context below.

If the answer is not available in the context,
clearly say that the paper does not provide
enough information to answer.

Use clear and simple language.
Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

                            # Generate the answer
                            response = llm.invoke(prompt)

                            st.subheader("AI Answer")
                            st.write(response.content)

                            # Display the retrieved source passages
                            st.subheader("Retrieved Sources")

                            for index, doc in enumerate(
                                relevant_docs, start=1
                            ):
                                page_number = (
                                    doc.metadata.get("page", 0) + 1
                                )

                                with st.expander(
                                    f"Source {index} — Page {page_number}"
                                ):
                                    st.write(doc.page_content)

                except Exception as error:
                    st.error(
                        f"Could not generate answer: {error}"
                    )