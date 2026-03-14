import os
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import config

class DocumentProcessor:
    def __init__(self):
        """
        Initializes the document processor setting up the embeddings and the local ChromaDB.
        """
        # 1. Setup Embeddings
        self.embeddings = HuggingFaceEmbeddings(
             model_name=config.EMBEDDING_MODEL_NAME,
             model_kwargs={"device": "cpu"}
)
        
        # 2. Setup Persistent Vector Database (Chroma)
        # By providing 'persist_directory', Chroma will automatically save
        # any ingested files to the hard drive, making them permanent.
        self.vector_store = Chroma(
            collection_name="ai_tutor_knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=config.DB_DIR
        )

        # 3. Setup Text Splitter
        # This breaks long PDF pages into smaller, meaningful chunks that the LLM can digest.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

    def process_and_store_txt(self, file_path: str) -> int:
        """
        Reads a TXT file, splits its text, and permanently stores the chunks in ChromaDB.
        Returns the number of chunks added.
        """
        print(f"Loading TXT document from: {file_path}")

        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read().strip()

        if not raw_text:
            return 0

        base_doc = Document(
            page_content=raw_text,
            metadata={"source": filename, "page": 1}
        )

        chunks: List[Document] = self.text_splitter.split_documents([base_doc])
        ids: List[str] = []
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            # Stable IDs prevent duplicate inserts when the same TXT is uploaded again.
            stable_source = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename.lower())
            ids.append(f"{stable_source}::chunk::{idx}")

        print(f"Extracted {len(chunks)} text chunks. Adding to Persistent Database...")
        self.vector_store.add_documents(chunks, ids=ids)
        print("Successfully stored in ChromaDB.")

        return len(chunks)

    def get_retriever(self, search_k: int = 4):
        """
        Returns a LangChain retriever object that can be used by the LLM
        to search the persistent database.
        """
        fetch_k = max(8, search_k * 2)
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": search_k, "fetch_k": fetch_k}
        )

    def retrieve_docs(self, query: str, k: int = 4) -> List[Document]:
        """
        Returns top-k relevant documents for direct evaluation/debugging use.
        """
        retriever = self.get_retriever(search_k=k)
        try:
            return retriever.get_relevant_documents(query)
        except Exception:
            return []

if __name__ == "__main__":
    # A simple test block to verify the database script works
    print("Testing database initialization...")
    dp = DocumentProcessor()
    print("Database connection and Embedding Model initialized successfully.")
