"""Provider-switchable chat model: OpenAI, Anthropic (Claude) or Google Gemini."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from copilot.config import get_settings

PROVIDERS = ("openai", "anthropic", "gemini")


def get_chat_model(provider: str | None = None, model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    s = get_settings()
    provider = (provider or s.llm_provider).lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or s.openai_model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or s.anthropic_model, temperature=temperature, max_tokens=1024)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or s.gemini_model, temperature=temperature)
    raise ValueError(f"Unknown provider '{provider}'. Choose from {PROVIDERS}.")
