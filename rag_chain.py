"""
rag_chain.py
------------
Step 2 of the pipeline: retrieval + generation, wired together with LangGraph.

Run this directly first, with hardcoded test questions, BEFORE touching the UI:
    python rag_chain.py

This is deliberately built as a graph (not a simple chain) since that's what
you're practicing for the interview -- it also makes it trivial to extend later
(e.g. add a "no relevant docs found" branch, or a web-search fallback node).
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()

VECTORSTORE_DIR = "vectorstore"

class RAGState(TypedDict):
    question: str
    context: List[str]      
    sources: List[str]     
    answer: str

def load_retriever(persist_dir: str = VECTORSTORE_DIR, k: int = 3):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


retriever = None


def get_retriever():
    global retriever
    if retriever is None:
        retriever = load_retriever()
    return retriever


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have enough information to answer that."
Cite which source each fact comes from using the [source] tags provided.

Match the depth of your answer to the question:
- If the user asks for a summary or a quick answer, keep it to 3-5 sentences.
- If the user asks for a detailed explanation, explanation, or "tell me everything",
  provide a thorough, comprehensive answer covering all relevant points from the context.
  Use bullet points, numbered lists, or paragraphs as appropriate.
- When in doubt, err on the side of being more detailed and thorough.

Do not invent information that is not in the context.
You are allowed to use your general knowledge to answer the question if the only when the user asks you to use it ,otherwise do not use it.

Context:
{context}

Question: {question}

Answer:"""
)



def retrieve_node(state: RAGState) -> RAGState:
    """Embed the question (implicitly, via the retriever) and fetch top-k chunks."""
    docs = get_retriever().invoke(state["question"])

    context = [doc.page_content for doc in docs]
    sources = [
        f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})"
        for doc in docs
    ]

    return {**state, "context": context, "sources": sources}


def generate_node(state: RAGState) -> RAGState:
    """Stuff retrieved chunks into the prompt and call the LLM."""
    context_str = "\n\n".join(
        f"[{src}]\n{chunk}" for chunk, src in zip(state["context"], state["sources"])
    )

    formatted_prompt = prompt.format(context=context_str, question=state["question"])
    response = llm.invoke(formatted_prompt)

    return {**state, "answer": response.content}


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_graph()


def ask(question: str) -> RAGState:
    """Convenience wrapper used by the Streamlit app."""
    result = rag_graph.invoke({"question": question, "context": [], "sources": [], "answer": ""})
    return result


if __name__ == "__main__":
    test_questions = [
        "What is this document about?",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        result = ask(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
