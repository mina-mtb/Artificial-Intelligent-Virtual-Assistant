import re

from openai import OpenAI, RateLimitError, AuthenticationError, APIConnectionError, OpenAIError
from langchain.memory import ConversationBufferMemory

import config


class TutorEngine:
    def __init__(self, retriever):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.retriever = retriever
        self.last_retrieved_sources = []

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def update_retriever(self, retriever):
        self.retriever = retriever

    def _build_chat_history(self) -> str:
        messages = self.memory.chat_memory.messages
        if not messages:
            return ""

        lines = []
        for msg in messages[-8:]:
            role = msg.type.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _retrieve_context(self, question: str) -> str:
        try:
            docs = self.retriever.get_relevant_documents(question)
        except Exception:
            docs = []

        self.last_retrieved_sources = []
        if not docs:
            return ""

        context_parts = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            key = (source, page, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            self.last_retrieved_sources.append(
                {"source": source, "page": page, "chunk_id": chunk_id}
            )
            context_parts.append(
                f"[Source: {source}, page {page}, chunk {chunk_id}]\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    def get_last_retrieved_sources(self):
        return self.last_retrieved_sources

    def _fallback_answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "No relevant context was found in uploaded TXT documents."

        question_terms = {
            t for t in re.findall(r"\w+", question.lower())
            if len(t) > 2
        }
        stop_terms = {"what", "when", "where", "which", "does", "about", "from", "with", "that", "this", "have", "your"}
        question_terms = {t for t in question_terms if t not in stop_terms}

        clean_lines = []
        for line in context.splitlines():
            if line.strip().startswith("[Source:"):
                continue
            if line.strip():
                clean_lines.append(line.strip())
        clean_text = " ".join(clean_lines)

        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        scored = []
        for sentence in sentences:
            terms = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(question_terms.intersection(terms))
            if overlap > 0:
                scored.append((overlap, sentence))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [s for _, s in scored[:2]]
            return "Based on the uploaded text, the answer is:\n\n" + " ".join(selected)

        return "I could not generate a concise answer, but relevant context was retrieved from your uploaded TXT document."

    def get_response(self, user_input: str, record_memory: bool = True) -> str:
        context = self._retrieve_context(user_input)
        chat_history = self._build_chat_history()

        if context.strip():
            prompt = f"""
You are a helpful, encouraging Socratic AI Tutor.

The user may ask greetings, simple conversational questions, or questions about uploaded study documents.

If the retrieved context is relevant, use it.
If the answer is not supported by the retrieved context, say that clearly.
Do not invent document-specific facts.

You can still respond naturally to greetings or general conversational messages.

Retrieved context:
{context}

Chat history:
{chat_history}

User message:
{user_input}
""".strip()
        else:
            prompt = f"""
You are a friendly and helpful AI Tutor.

Right now there is no relevant uploaded document context available.
So:
- respond naturally to greetings and normal conversation
- help the user in a useful way
- if the user asks about uploaded documents, explain that no relevant document context is available yet
- do not pretend that you have read documents when you have not

Chat history:
{chat_history}

User message:
{user_input}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful, friendly tutor."},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
        except RateLimitError:
            answer = self._fallback_answer(user_input, context)
        except AuthenticationError:
            answer = self._fallback_answer(user_input, context)
        except APIConnectionError:
            answer = self._fallback_answer(user_input, context)
        except OpenAIError:
            answer = self._fallback_answer(user_input, context)

        if record_memory:
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(answer)

        return answer

    def generate_quiz(self) -> str:
        context = self._retrieve_context(
            "Generate a quiz from the uploaded study materials."
        )

        if not context.strip():
            return "I do not have enough uploaded study material yet to generate a contextual quiz. Please upload at least one PDF first."

        prompt = f"""
You are a helpful academic tutor.

Based only on the study material context below, generate a 3-question multiple-choice quiz.
Each question should have 4 options.
At the end, include the correct answers.

Study material context:
{context}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful academic tutor."},
                    {"role": "user", "content": prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError:
            return "OpenAI API quota is exceeded (429). Please check billing/quota and try again."
        except AuthenticationError:
            return "OpenAI API key is invalid or missing. Please update OPENAI_API_KEY in .env."
        except APIConnectionError:
            return "Could not connect to OpenAI. Please check internet/network settings and retry."
        except OpenAIError:
            return "OpenAI request failed. Please try again in a moment."
