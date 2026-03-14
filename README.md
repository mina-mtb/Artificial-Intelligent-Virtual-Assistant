# Socratic AI Tutor (RAG-Based Virtual Assistant)

This project is a multi-file educational assistant built with Streamlit, OpenAI, and ChromaDB.
It supports Retrieval-Augmented Generation (RAG) over uploaded study files and includes a built-in evaluation panel.

## What This App Does

- Upload and save study files (`txt`, `md`, `pdf`, `csv`, `json`)
- Build persistent vector memory from uploaded content
- Chat with grounded answers from saved files
- Show retrieved evidence (source/page/chunk) for traceability
- Evaluate retrieval + generation quality with a golden dataset
- Delete one saved file or reset all saved knowledge

## Tech Stack

- UI: Streamlit
- LLM: OpenAI Chat Completions API
- Embeddings: Sentence-Transformers (`all-MiniLM-L6-v2`)
- Vector DB: Chroma (persistent)
- Parsing: PyPDF2 + text/csv/json loaders
- Containerization: Docker Compose

## Project Structure

- `app.py`: Main Streamlit app (left knowledge, center chat, right assessment)
- `database.py`: File ingestion, chunking, vector store operations, delete/reset logic
- `tutor_engine.py`: Response generation, grounding checks, chitchat handling
- `evaluation.py`: Hit@k, MRR, F1, semantic similarity metrics
- `config.py`: Runtime settings
- `docker-compose.yml` + `Dockerfile`: Containerized run

## Prerequisites

- Docker Desktop (recommended) OR Python 3.11+
- OpenAI API key

## Environment Variables

Create `.env` in project root:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_OUTPUT_TOKENS=220
```

## Run With Docker (Recommended)

1. Clone repo and enter folder
```bash
git clone <YOUR_REPO_URL>
cd "Artificial Intelligent Virtual Assistant"
```

2. Build and run
```bash
docker compose up -d --build
```

3. Open app
```text
http://localhost:8501
```

4. Stop app
```bash
docker compose down
```

## Run Locally (Without Docker)

1. Create and activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Start app
```bash
streamlit run app.py
```

## First-Run and Persistence Behavior

- On first run, saved knowledge starts empty.
- After saving files, knowledge persists across restarts.
- Saved memory can be managed from the UI:
  - delete per file (`X` button)
  - delete all (`Reset Knowledge Base`)

## How to Use

1. In **Knowledge Base** (left panel), upload a file.
2. Click **Process & Save Document**.
3. Confirm file appears under **Saved Files**.
4. Ask questions in **Chat** (center panel).
5. Use **Assessment** (right panel) after at least one saved file.

## Evaluation

The app can evaluate on `golden_dataset.jsonl` and reports:

- `Hit@1`, `Hit@3`
- `MRR`
- `Answer F1 (avg)`
- `Semantic Similarity (avg)`

## Troubleshooting

- If API errors occur, verify `OPENAI_API_KEY` in `.env`.
- If UI seems stale after changes, hard refresh browser (`Ctrl+F5`).
- If Docker storage becomes full:
```bash
docker system prune -af --volumes
```

## License

For academic/project use. Add a formal license file if publishing publicly.
