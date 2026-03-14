import os
import re
import csv
import json
from typing import Dict, List
from PyPDF2 import PdfReader
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

    def _read_txt_like(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    def _read_pdf(self, file_path: str) -> str:
        text_parts: List[str] = []
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    text_parts.append(page_text)
        except Exception:
            return ""
        return "\n\n".join(text_parts).strip()

    def _read_csv(self, file_path: str) -> str:
        lines: List[str] = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    clean_row = [c.strip() for c in row if c and c.strip()]
                    if clean_row:
                        lines.append(" | ".join(clean_row))
        except Exception:
            return ""
        return "\n".join(lines).strip()

    def _flatten_json(self, obj, prefix: str = "") -> List[str]:
        lines: List[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                lines.extend(self._flatten_json(v, key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                key = f"{prefix}[{i}]"
                lines.extend(self._flatten_json(item, key))
        else:
            value = str(obj).strip()
            if value:
                lines.append(f"{prefix}: {value}" if prefix else value)
        return lines

    def _read_json(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            return ""
        return "\n".join(self._flatten_json(data)).strip()

    def _load_file_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".txt", ".md"}:
            return self._read_txt_like(file_path)
        if ext == ".pdf":
            return self._read_pdf(file_path)
        if ext == ".csv":
            return self._read_csv(file_path)
        if ext == ".json":
            return self._read_json(file_path)
        return ""

    def process_and_store_file(self, file_path: str) -> int:
        """
        Reads a supported file, splits its text, and permanently stores the chunks in ChromaDB.
        Returns the number of chunks added.
        """
        print(f"Loading document from: {file_path}")

        filename = os.path.basename(file_path)
        raw_text = self._load_file_text(file_path)

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

    def _tokenize(self, text: str) -> set:
        return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}

    def _hybrid_rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []

        q_tokens = self._tokenize(query)
        scored = []
        for rank, doc in enumerate(docs, start=1):
            d_tokens = self._tokenize(doc.page_content)
            lexical = 0.0
            if q_tokens and d_tokens:
                lexical = len(q_tokens.intersection(d_tokens)) / max(1, len(q_tokens))

            semantic = 1.0 / rank
            score = 0.7 * semantic + 0.3 * lexical
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        unique_docs: List[Document] = []
        seen = set()
        for _, doc in scored:
            source = doc.metadata.get("source", "")
            chunk_id = doc.metadata.get("chunk_id", "")
            key = (source, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
            if len(unique_docs) >= top_k:
                break
        return unique_docs

    def get_retriever(self, search_k: int = None):
        """
        Returns a LangChain retriever object that can be used by the LLM
        to search the persistent database.
        """
        if search_k is None:
            search_k = config.DEFAULT_RETRIEVAL_K
        fetch_k = max(8, search_k * config.RETRIEVAL_FETCH_K_FACTOR)
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": search_k, "fetch_k": fetch_k}
        )

    def retrieve_docs(self, query: str, k: int = None) -> List[Document]:
        """
        Returns top-k relevant documents for direct evaluation/debugging use.
        """
        if k is None:
            k = config.DEFAULT_RETRIEVAL_K
        fetch_k = max(8, k * config.RETRIEVAL_FETCH_K_FACTOR)
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k}
        )
        try:
            docs = retriever.get_relevant_documents(query)
            return self._hybrid_rerank(query, docs, top_k=k)
        except Exception:
            return []

    def list_saved_files(self) -> List[Dict[str, int]]:
        """
        Returns a list of saved sources currently present in vector storage.
        Each item includes filename and chunk_count.
        """
        try:
            payload = self.vector_store.get(include=["metadatas"])
        except Exception:
            return []

        metadatas = payload.get("metadatas", []) if isinstance(payload, dict) else []
        counts: Dict[str, int] = {}
        for md in metadatas:
            if not isinstance(md, dict):
                continue
            source = str(md.get("source", "")).strip()
            if not source:
                continue
            counts[source] = counts.get(source, 0) + 1

        rows = [{"source": k, "chunk_count": v} for k, v in counts.items()]
        rows.sort(key=lambda x: x["source"].lower())
        return rows

    def _ids_for_source(self, source: str) -> List[str]:
        try:
            payload = self.vector_store.get(include=["metadatas"])
        except Exception:
            return []

        if not isinstance(payload, dict):
            return []
        ids = payload.get("ids", []) or []
        metadatas = payload.get("metadatas", []) or []
        matched_ids: List[str] = []
        for i, md in enumerate(metadatas):
            if not isinstance(md, dict):
                continue
            if str(md.get("source", "")).strip() == source and i < len(ids):
                matched_ids.append(ids[i])
        return matched_ids

    def delete_source(self, source: str) -> int:
        """
        Deletes all chunks that belong to a given source filename.
        Returns the number of deleted chunks.
        """
        ids = self._ids_for_source(source)
        if not ids:
            return 0
        try:
            self.vector_store.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def clear_all_documents(self) -> int:
        """
        Deletes all chunks from the current vector collection.
        Returns the number of deleted chunks.
        """
        try:
            payload = self.vector_store.get(include=[])
        except Exception:
            return 0

        if not isinstance(payload, dict):
            return 0
        ids = payload.get("ids", []) or []
        if not ids:
            return 0
        try:
            self.vector_store.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

if __name__ == "__main__":
    # A simple test block to verify the database script works
    print("Testing database initialization...")
    dp = DocumentProcessor()
    print("Database connection and Embedding Model initialized successfully.")
