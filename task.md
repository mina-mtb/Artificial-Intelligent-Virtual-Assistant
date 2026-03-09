# AI Tutor Project Tasks

## Phase 0: Continuous Documentation & Tutorial
- [ ] **0.1 Initialize Project Tutorial Document**
  - [ ] Set up a Markdown/LaTeX source file that will eventually be compiled into the final PDF tutorial.
- [ ] **0.2 Continuous Tutorial Updates**
  - [ ] After each successful implementation step (setup, backend, UI, voice, docker), write a detailed section explaining *how* and *why* it was done.
  - [ ] Compile the tutorial into a PDF format regularly to track progress.

## Phase 1: Foundation & Persistent Database
- [ ] **1.1 Setup Project Environment**
  - [ ] Create `requirements.txt` (LangChain, ChromaDB, Streamlit, PyPDF2, Sentence-Transformers).
  - [ ] Set up project directory structure and basic configuration (`config.py`).
- [ ] **1.2 Document Processing & Persistent Vector Storage**
  - [ ] Implement PDF text extraction using PyPDF2.
  - [ ] Implement intelligent document chunking (`RecursiveCharacterTextSplitter`).
  - [ ] Configure local `sentence-transformers` for embeddings.
  - [ ] Setup ChromaDB with **Persistent Storage** enabled (to keep files like old and new blood tests permanently).

## Phase 2: Core RAG & Conversational Memory
- [ ] **2.1 Retrieval System**
  - [ ] Create retriever pipeline from the persistent ChromaDB to fetch relevant context based on user queries across all stored documents.
- [ ] **2.2 LLM Integration (Ollama/Llama 3)**
  - [ ] Connect LangChain to the local Ollama instance running the chosen Llama model.
- [ ] **2.3 Conversation Memory**
  - [ ] Implement `ConversationBufferMemory` to track dialogue history for coherent multi-turn interactions.

## Phase 3: AI Tutor Logic & Socratic Persona
- [ ] **3.1 Socratic Prompt Engineering**
  - [ ] Design System Prompts to enforce the "Socratic Tutor" persona (guiding students rather than just giving answers).
- [ ] **3.2 Adaptive Responses & Comprehension Tracking**
  - [ ] Implement logic to adapt the complexity of the response based on the conversation history and user skill level.
- [ ] **3.3 Contextual Quiz Generation**
  - [ ] Build a dedicated LangChain agent/chain that creates multiple-choice or short-answer quizzes based on the stored PDF knowledge.

## Phase 4: Streamlit Frontend Interaction
- [ ] **4.1 UI Layout**
  - [ ] Create a Streamlit interface with a sidebar for document uploads and a main area for the chat interface.
- [ ] **4.2 Connecting UI to Backend**
  - [ ] Wire up document uploads to trigger the document ingestion pipeline and save to the persistent database.
  - [ ] Wire up the chat input to the RAG memory chain.
  - [ ] Add functional buttons for "Generate Quiz" to interact with the Quiz Generation chain.
- [ ] **4.3 Voice Interaction Integration (Speech-to-Text & Text-to-Speech)**
  - [ ] Integrate a Speech-to-Text (STT) library (e.g., `SpeechRecognition` or `whisper`) to capture user voice input directly in the browser/app.
  - [ ] Integrate a Text-to-Speech (TTS) library (e.g., `gTTS` or `pyttsx3`) or browser native synthesis to read Llama's responses aloud.
  - [ ] Add UI controls (microphone icon, speaker icon) to toggle voice modes on and off.

## Phase 5: Deployment & Containerization
- [ ] **5.1 Docker Configuration**
  - [ ] Write a `Dockerfile` to containerize the Streamlit web application and its dependencies.
- [ ] **5.2 Docker Compose Orchestration**
  - [ ] Create a `docker-compose.yml` to orchestrate the Streamlit app container and map persistent volumes for ChromaDB so data isn't lost on restart.
  - [ ] Configure networking so the container can securely access the host's Ollama instance (since running LLMs inside containers requires complex GPU mapping, it's safer to connect the containerized app to the local/host Ollama).
  - [ ] Test the full deployment process (`docker-compose up`).
