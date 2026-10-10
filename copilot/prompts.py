"""Prompt text, kept free of heavy imports so it can be reused (RAG, evaluation, DPO data)."""

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are an enterprise knowledge copilot. Answer ONLY from the numbered context passages.\n"
    "Rules:\n"
    "1. Cite the passages you used like [1] or [2][3] right after each claim.\n"
    "2. If the context does not contain the answer, reply exactly: "
    '"I don\'t know based on the provided documents."\n'
    "3. Be concise and never invent facts, numbers, names or dates."
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


def format_context(docs) -> str:
    """docs: objects with .page_content and .metadata (LangChain Documents)."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        blocks.append(f"[{i}] ({m.get('source')}, {m.get('modality')}, {m.get('locator')})\n{d.page_content}")
    return "\n\n".join(blocks)


def build_user_prompt(context: str, question: str) -> str:
    return HUMAN_TEMPLATE.format(context=context, question=question)
