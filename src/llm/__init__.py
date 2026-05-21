from .base import BaseLLM


def get_llm(provider: str = "gemini") -> BaseLLM:
    provider = provider.lower()

    if provider == "gemini":
        from .gemini import GeminiLLM
        return GeminiLLM()

    elif provider == "claude":
        from .claude import ClaudeLLM
        return ClaudeLLM()


    raise ValueError(f"Unsupported LLM provider: {provider}")