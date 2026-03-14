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
            return "No relevant context was found in uploaded study files."

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

        return "I could not generate a concise answer, but relevant context was retrieved from your uploaded file."

    def _trim_to_two_sentences(self, text: str) -> str:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return text.strip()
        return " ".join(parts[:2]).strip()

    def _extract_clean_context(self, context: str) -> str:
        lines = []
        for line in context.splitlines():
            if line.strip().startswith("[Source:"):
                continue
            if line.strip():
                lines.append(line.strip())
        return " ".join(lines).strip()

    def _extractive_answer(self, question: str, context: str) -> str:
        clean_context = self._extract_clean_context(context)
        if not clean_context:
            return ""

        question_terms = {
            t for t in re.findall(r"\w+", question.lower())
            if len(t) > 2
        }
        stop_terms = {
            "what", "when", "where", "which", "does", "about", "from", "with",
            "that", "this", "have", "your", "give", "common", "used", "mean",
            "important", "why", "is", "the", "are", "for", "and"
        }
        question_terms = {t for t in question_terms if t not in stop_terms}
        if not question_terms:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", clean_context)
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_terms = {t for t in re.findall(r"\w+", sent.lower()) if len(t) > 2}
            if not sent_terms:
                continue
            overlap = len(question_terms.intersection(sent_terms))
            if overlap == 0:
                continue
            score = overlap / max(1, len(question_terms))
            scored.append((score, sent))

        if not scored:
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sentence = scored[0]
        if best_score < 0.20:
            return ""
        return self._trim_to_two_sentences(best_sentence)

    def _grounding_strength(self, question: str, context: str) -> float:
        clean_context = self._extract_clean_context(context)
        if not clean_context:
            return 0.0
        q_terms = {t for t in re.findall(r"\w+", question.lower()) if len(t) > 2}
        c_terms = {t for t in re.findall(r"\w+", clean_context.lower()) if len(t) > 2}
        if not q_terms or not c_terms:
            return 0.0
        overlap = len(q_terms.intersection(c_terms))
        return overlap / max(1, len(q_terms))

    def _contains_persian(self, text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text))

    def _is_chitchat(self, text: str) -> bool:
        t = text.strip().lower()
        if not t:
            return True

        normalized = re.sub(r"[^\w\u0600-\u06FF\s]", " ", t)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        en_smalltalk = {
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye",
            "how are you", "good morning", "good evening", "good night"
        }
        fa_smalltalk = {
            "سلام", "درود", "مرسی", "ممنون", "خوبی", "خداحافظ", "شب بخیر", "صبح بخیر", "عصر بخیر"
        }

        if normalized in en_smalltalk or normalized in fa_smalltalk:
            return True
        if len(normalized.split()) <= 2 and normalized in fa_smalltalk:
            return True
        if len(normalized.split()) <= 3 and normalized in en_smalltalk:
            return True
        return False

    def _chitchat_answer(self, text: str) -> str:
        t = text.strip().lower()
        is_fa = self._contains_persian(text)
        if is_fa:
            if any(w in t for w in ["خداحافظ"]):
                return "خداحافظ. هر وقت خواستی دوباره ادامه می‌دهیم."
            if any(w in t for w in ["مرسی", "ممنون"]):
                return "خواهش می‌کنم. اگر خواستی، سوال بعدی را بپرس."
            return "سلام. خوشحالم اینجا هستی. اگر بخواهی می‌توانم روی فایل‌های آپلودشده کمکت کنم."

        if any(w in t for w in ["bye", "goodbye"]):
            return "Goodbye. I am here whenever you want to continue."
        if any(w in t for w in ["thanks", "thank you"]):
            return "You're welcome. Ask me anything when you're ready."
        return "Hello. I'm ready to help with your uploaded study files whenever you want."

    def get_response(self, user_input: str, record_memory: bool = True) -> str:
        if self._is_chitchat(user_input):
            self.last_retrieved_sources = []
            answer = self._chitchat_answer(user_input)
            if record_memory:
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(answer)
            return answer

        context = self._retrieve_context(user_input)
        chat_history = self._build_chat_history()
        grounding = self._grounding_strength(user_input, context)

        if context.strip():
            extractive = self._extractive_answer(user_input, context)
            if extractive:
                answer = extractive
                if record_memory:
                    self.memory.chat_memory.add_user_message(user_input)
                    self.memory.chat_memory.add_ai_message(answer)
                return answer

        if context.strip():
            if grounding < 0.18:
                answer = (
                    "I don't have enough evidence in the retrieved context to answer confidently. "
                    "Please upload a more relevant file or rephrase the question with key course terms."
                )
                if record_memory:
                    self.memory.chat_memory.add_user_message(user_input)
                    self.memory.chat_memory.add_ai_message(answer)
                return answer

            prompt = f"""
You are a helpful, encouraging Socratic AI Tutor.

Follow these rules strictly:
1) Use only the retrieved context for document-grounded answers.
2) If the answer is not clearly supported by context, say: "I don't have enough evidence in the retrieved context."
3) Keep the answer concise (1-2 short sentences).
4) Reuse wording from retrieved context where possible.
5) Do not add unrelated details.
6) If the user message is a greeting/chitchat, respond naturally and briefly.
7) If evidence is partial, explicitly state uncertainty and mention what is missing.

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

Right now there is no relevant uploaded study-file context available.
So:
- respond naturally to greetings and normal conversation
- help the user in a useful way
- if the user asks about uploaded files, explain that no relevant file context is available yet
- do not pretend that you have read files when you have not

Chat history:
{chat_history}

User message:
{user_input}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_OUTPUT_TOKENS,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise educational tutor. "
                            "When context exists, answer with minimal, direct wording and no hallucination."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            answer = self._trim_to_two_sentences((response.choices[0].message.content or "").strip())
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
