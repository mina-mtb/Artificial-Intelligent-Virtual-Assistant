# Implementation Plan: AI Tutor Web App

This project transforms static PDF documents into an interactive learning companion using Retrieval-Augmented Generation (RAG). It fulfills all requirements from the Group 77 Proposal while integrating the user's specific request for persistent knowledge storage.

## Core Objectives Covered
1. **Persistent Knowledge Base**: (User Request) Documents uploaded (e.g., month 1 blood test vs month 2 blood test) remain in the system's "long-term memory" and can be cross-referenced or compared at any time.
2. **Socratic Tutor Persona**: (Proposal) The Agent will actively engage the user in Socratic dialogue, asking guiding questions rather than just feeding answers.
3. **Contextual Quiz Generation**: (Proposal) Capability to generate quizzes to test student comprehension based on uploaded material.
4. **Adaptive Conversation Memory**: (Proposal) Managing multi-turn interaction to keep context coherent using LangChain memory buffers.
5. **Continuous Tutorial Documentation**: (User Request) A living PDF document will be maintained that explains the architecture, setup process, and code explanations step-by-step, acting as a comprehensive tutorial for the entire project.

## Technical Architecture

### 1. Persistent Storage Layer (ChromaDB)
- **Component**: `database.py`
- **Function**: Handles ingestion using PyPDF2, text splitting via `RecursiveCharacterTextSplitter`, and embedding generation using local `Sentence-Transformers`. 
- **Crucial Feature**: Configured with a `persist_directory`. Data is never lost when the app reloads. This directly solves the requirement to compare old and new medical data or course notes.

### 2. LLM Engine (Ollama & LangChain)
- **Component**: `tutor_engine.py`
- **Function**: Connects the persistent retriever to the local Ollama instance (e.g., Llama 3).
- **Sub-Chains**:
    - **QA/Socratic Chain**: Handles standard user queries with a heavily tuned System Prompt enforcing the teaching persona.
    - **Quiz Chain**: A separate LLM call specifically prompted to extract facts from retrieved documents and format them into interactive quiz questions.
    - **Memory**: Utilizes `ConversationBufferMemory` to pass previous chat history into the LLM context.

### 3. Audio Processing Layer (Voice Interaction)
- **Component**: `audio_processor.py` (New addition based on user request)
- **Function**: Enables ChatGPT-like voice conversation capabilities.
- **Workflow**:
    - **Speech-to-Text (STT)**: Captures user microphone input via Streamlit audio recording components and transcribes it to text (using libraries like `whisper` or standard `SpeechRecognition`). This text is then passed to the LLM Engine just like typed text.
    - **Text-to-Speech (TTS)**: Takes the generated text response from Ollama/Llama and converts it back into an audio file (using tools like `gTTS` or local `pyttsx3`) that automatically plays back to the user.

### 4. Streamlit Frontend
- **Component**: `app.py`
- **Function**: The web interface replacing the complicated mobile requirement with a simple, browser-based UI.
- **Workflow**: 
    - Sidebar: Upload PDFs -> Triggers ingestion into Persistent ChromaDB.
    - Main Panel: ChatGPT-style interface for the Socratic Tutor.
    - UI Actions: A "Start Quiz" button that shifts the LLM chain to generate and grade questions.

### 5. Deployment Layer (Docker)
- **Component**: `Dockerfile` and `docker-compose.yml`
- **Function**: Containerizes the Web App and Database for production-ready deployment on any server.
- **Strategy**: 
    - The Streamlit App, Audio Processor, and ChromaDB are packaged into a single Docker image.
    - `docker-compose` mounts a local directory as a volume to ensure ChromaDB's persistent data survives container restarts.
    - The container is configured to communicate with the host machine's Ollama instance (to leverage the host's GPU/CPU effectively without complex container GPU-passthrough setups).

## Verification Plan
1. **Persistence Test**: Upload Document A. Close app. Open app. Ask about Document A -> Must answer correctly. 
2. **Comparison Test**: Upload Document A (Month 1 test) and Document B (Month 2 test). Ask "What changed between month 1 and 2?" -> Must retrieve both and compare.
3. **Socratic Test**: Ask "What is X?" -> Tutor must prompt back or guide, rather than just copy-pasting the definition of X.
4. **Quiz Test**: Click "Generate Quiz" -> Must output valid questions based *only* on the ingested PDF context.
