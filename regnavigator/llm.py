from typing import List, Dict
from .llm_providers import get_provider

SYSTEM_PROMPT = """You are a careful compliance analyst.
Answer ONLY using the information in the provided snippets.
If the answer is not present, say "Not found in provided sources."
When you use a snippet, cite it with bracket numbers like [1], [2].
Keep answers concise and precise."""

USER_TEMPLATE = """Question: {question}

Snippets:
{snippets}

Instructions:
- Only use these snippets
- Cite snippet numbers like [1][2]
- If unsure, respond: "Not found in provided sources." """

def build_snippet_block(chunks: List[Dict]) -> str:
    lines = []
    for i, ch in enumerate(chunks, start=1):
        m = ch.get("meta", {})
        header_str = f" — {m.get('header')}" if m.get('header') else ""
        meta_line = f"(File: {m.get('source_file')}, Page: {m.get('page')}{header_str})"
        txt = ch["text"].strip()
        if len(txt) > 1200:
            txt = txt[:1200] + "…"
        lines.append(f"[{i}] {txt}\n{meta_line}")
    return "\n\n".join(lines)

class LLM:
    def __init__(self, provider_name: str = None, model: str = None):
        self.provider = get_provider(provider_name, model)
    
    def available(self) -> bool:
        return self.provider.available()
    
    def answer(self, question: str, chunks: List[Dict]) -> str:
        if not chunks:
            return "Not found in provided sources."
        user = USER_TEMPLATE.format(question=question, snippets=build_snippet_block(chunks))
        return self.provider.chat(system=SYSTEM_PROMPT, user=user, temperature=0.1)
