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
        retriever=st.session_state.doc_processor.get_retriever()
    )
if "audio_processor" not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Socratic AI Tutor. Ask me any question about the uploaded TXT documents!",
        }
    ]

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
        st.dataframe(eval_result["rows"], use_container_width=True)

with st.sidebar:
    st.header("Knowledge Base")
    st.write("Upload TXT files to add them to the AI's memory.")

    uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])
    if st.button("Process & Save Document") and uploaded_file:
        with st.spinner("Processing document..."):
            temp_path = os.path.join(config.TEMP_UPLOAD_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            num_chunks = st.session_state.doc_processor.process_and_store_txt(temp_path)
            st.session_state.tutor_engine.update_retriever(
                st.session_state.doc_processor.get_retriever()
            )

            if num_chunks == 0:
                st.warning("TXT file was empty. Nothing was added.")
            else:
                st.success(
                    f"TXT processed successfully! ({num_chunks} chunks added to persistent storage)"
                )

    st.divider()
    st.header("Assessment")
    st.button(
        "Generate Contextual Quiz",
        disabled=True,
        help="Coming soon (outside MVP scope).",
    )

    dataset_path = os.path.join(config.BASE_DIR, "golden_dataset.jsonl")
    if st.button("Evaluate Model"):
        with st.spinner("Running retrieval/generation evaluation..."):
            dataset = load_golden_dataset(dataset_path)
            result = evaluate_dataset(
                doc_processor=st.session_state.doc_processor,
                tutor_engine=st.session_state.tutor_engine,
                dataset=dataset,
                k_values=(1, 3, 5),
            )
            st.session_state.eval_result = result
    with st.popover("View Evaluation Results"):
        render_evaluation_results()

    st.divider()
    st.write(
        "Tip: Because of persistent storage, you don't need to re-upload files unless they are new."
    )

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

user_input = st.chat_input("Ask a question...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = st.session_state.tutor_engine.get_response(user_input)
            st.write(response_text)
            current_sources = st.session_state.tutor_engine.get_last_retrieved_sources()
            if current_sources:
                source_lines = []
                for s in current_sources:
                    source_lines.append(
                        f"- source: `{s.get('source', 'unknown')}`, page: `{s.get('page', '?')}`, chunk: `{s.get('chunk_id', '?')}`"
                    )
                st.markdown("Retrieved evidence:\n" + "\n".join(source_lines))

            audio_html = st.session_state.audio_processor.text_to_speech_html(response_text)
            st.markdown(audio_html, unsafe_allow_html=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": st.session_state.tutor_engine.get_last_retrieved_sources(),
        }
    )
