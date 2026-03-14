import os
import streamlit as st

import config
from audio_processor import AudioProcessor
from database import DocumentProcessor
from evaluation import evaluate_dataset, load_golden_dataset
from tutor_engine import TutorEngine


if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()
if "tutor_engine" not in st.session_state:
    st.session_state.tutor_engine = TutorEngine(
        retriever=st.session_state.doc_processor.get_retriever(
            search_k=config.DEFAULT_RETRIEVAL_K
        )
    )
if "audio_processor" not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Socratic AI Tutor. Ask me any question about your uploaded study files.",
        }
    ]
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = config.DEFAULT_RETRIEVAL_K

st.set_page_config(page_title="Socratic AI Tutor", page_icon="🎓", layout="wide")
st.title("👨‍🏫 Socratic AI Tutor with Persistent Memory")

def render_evaluation_results():
    eval_result = st.session_state.get("eval_result")
    if not eval_result:
        st.info("No evaluation result yet. Click 'Evaluate Model' first.")
        return

    if eval_result["num_samples"] == 0:
        st.info(
            "No golden dataset found. Create 'golden_dataset.jsonl' in the project root to run evaluation."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", eval_result["num_samples"])
        c2.metric("Hit@1", eval_result["hit_rates"]["Hit@1"])
        c3.metric("Hit@3", eval_result["hit_rates"]["Hit@3"])
        c4.metric("MRR", eval_result["mrr"])
        st.metric("Answer F1 (avg)", eval_result["avg_f1"])
        st.metric("Semantic Similarity (avg)", eval_result.get("avg_semantic_similarity", 0.0))
        st.dataframe(eval_result["rows"], use_container_width=True, height=360)

def render_saved_files_panel():
    st.subheader("Saved Files")
    saved_files = st.session_state.doc_processor.list_saved_files()
    if not saved_files:
        st.caption("No files saved yet.")
        return
    for row in saved_files:
        c1, c2 = st.columns([0.88, 0.12], gap="small")
        c1.markdown(f"`{row['source']}`  ({row['chunk_count']} chunks)")
        if c2.button("X", key=f"delete_{row['source']}", help="Delete this file", use_container_width=True):
            deleted = st.session_state.doc_processor.delete_source(row["source"])
            st.session_state.tutor_engine.update_retriever(
                st.session_state.doc_processor.get_retriever(
                    search_k=st.session_state.retrieval_k
                )
            )
            if deleted > 0:
                st.success(f"Deleted `{row['source']}` ({deleted} chunks)")
            else:
                st.warning(f"Could not delete `{row['source']}`")
            st.rerun()

left_col, center_col, right_col = st.columns([1.0, 1.8, 1.0], gap="large")

with left_col:
    st.header("Knowledge Base")
    st.write("Upload study files, save them, and see what is currently stored.")

    retrieval_k = st.slider(
        "Retrieval Top-k",
        min_value=1,
        max_value=8,
        value=st.session_state.retrieval_k,
        help="Lower values are stricter; higher values provide broader context.",
        key="left_retrieval_k",
    )
    if retrieval_k != st.session_state.retrieval_k:
        st.session_state.retrieval_k = retrieval_k
        st.session_state.tutor_engine.update_retriever(
            st.session_state.doc_processor.get_retriever(
                search_k=st.session_state.retrieval_k
            )
        )

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["txt", "md", "pdf", "csv", "json"],
        key="left_file_uploader",
    )
    if st.button("Process & Save Document", key="left_save_btn") and uploaded_file:
        with st.spinner("Processing document..."):
            temp_path = os.path.join(config.TEMP_UPLOAD_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            num_chunks = st.session_state.doc_processor.process_and_store_file(temp_path)
            st.session_state.tutor_engine.update_retriever(
                st.session_state.doc_processor.get_retriever(
                    search_k=st.session_state.retrieval_k
                )
            )

            if num_chunks == 0:
                st.warning("File was empty or unsupported. Nothing was added.")
            else:
                st.success(f"Saved: `{uploaded_file.name}` ({num_chunks} chunks)")
                st.rerun()

    st.divider()
    if st.button("Reset Knowledge Base (Delete All)", key="left_reset_all_btn", type="secondary"):
        deleted_total = st.session_state.doc_processor.clear_all_documents()
        st.session_state.tutor_engine.update_retriever(
            st.session_state.doc_processor.get_retriever(
                search_k=st.session_state.retrieval_k
            )
        )
        if deleted_total > 0:
            st.success(f"All saved knowledge deleted ({deleted_total} chunks).")
        else:
            st.info("Knowledge base was already empty.")
        st.rerun()

    render_saved_files_panel()

with center_col:
    st.header("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                source_lines = []
                for s in msg["sources"]:
                    source_lines.append(
                        f"- source: `{s.get('source', 'unknown')}`, page: `{s.get('page', '?')}`, chunk: `{s.get('chunk_id', '?')}`"
                    )
                st.markdown("Retrieved evidence:\n" + "\n".join(source_lines))

    st.divider()
    with st.form("center_chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([0.86, 0.14], gap="small")
        user_input = input_col.text_input(
            "Message",
            key="center_chat_input",
            placeholder="Ask a question...",
            label_visibility="collapsed",
        )
        send_clicked = send_col.form_submit_button("Send", use_container_width=True)

    if send_clicked and user_input.strip():
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            response_text = st.session_state.tutor_engine.get_response(user_input.strip())
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "sources": st.session_state.tutor_engine.get_last_retrieved_sources(),
                }
            )
        st.rerun()

with right_col:
    st.header("Assessment")
    has_saved_files = len(st.session_state.doc_processor.list_saved_files()) > 0
    st.button(
        "Generate Contextual Quiz",
        disabled=True,
        help="Coming soon (outside MVP scope).",
        key="right_quiz_btn",
    )

    dataset_path = os.path.join(config.BASE_DIR, "golden_dataset.jsonl")
    if st.button(
        "Evaluate Model",
        key="right_eval_btn",
        disabled=not has_saved_files,
        help="Save at least one file first." if not has_saved_files else None,
    ):
        with st.spinner("Running retrieval/generation evaluation..."):
            dataset = load_golden_dataset(dataset_path)
            result = evaluate_dataset(
                doc_processor=st.session_state.doc_processor,
                tutor_engine=st.session_state.tutor_engine,
                dataset=dataset,
                k_values=(1, 3, 5),
            )
            st.session_state.eval_result = result
    with st.expander("View Evaluation Results", expanded=False):
        render_evaluation_results()
