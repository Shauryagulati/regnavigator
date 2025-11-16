import os
from anthropic import Anthropic
from openai import OpenAI
from .config import LLM_PROVIDER, LLM_MODEL

class BaseProvider:
    def available(self):
        return True

    def chat(self, system, user, temperature=0.0):
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            self._available = False
            return
        self.client = OpenAI(api_key=key)
        self._available = True

    def available(self):
        return self._available

    def chat(self, system, user, temperature=0.0):
        out = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )

        return out.choices[0].message.content


class AnthropicProvider(BaseProvider):
    def __init__(self):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            self._available = False
            return
        self.client = Anthropic(api_key=key)
        self._available = True

    def available(self):
        return self._available

    def chat(self, system, user, temperature=0.0):
        msg = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature
        )
        return msg.content[0].text


def get_provider(name=None, model=None):
    name = name or LLM_PROVIDER

    if name.lower() == "openai":
        return OpenAIProvider()
    return AnthropicProvider()

def validate_llm_config():
    """
    Simple helper used by check_system to verify that LLM configuration
    looks usable. It does NOT actually call the provider, just checks env vars.
    Returns a dict like:
    {
        "ready": bool,
        "provider": "...",
        "model": "...",
        "issues": [str, ...]
    }
    """
    provider = (LLM_PROVIDER or "openai").lower()
    issues = []
    ready = True

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key.startswith("sk-proj-xxx"):
            ready = False
            issues.append("OPENAI_API_KEY missing or placeholder")
    elif provider in ("claude", "anthropic"):
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key or key.startswith("sk-ant-xxx"):
            ready = False
            issues.append("ANTHROPIC_API_KEY missing or placeholder")
    else:
        ready = False
        issues.append(f"Unknown LLM_PROVIDER '{provider}'")

    return {
        "ready": ready,
        "provider": provider,
        "model": LLM_MODEL,
        "issues": issues,
    }
