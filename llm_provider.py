"""
llm_provider.py — centralised factory functions for LLM and embedding clients.

All agents and tools MUST use get_llm() and get_embeddings() from here.
Never instantiate ChatOpenAI, ChatAnthropic, HuggingFaceEmbeddings, etc.
directly anywhere else in the codebase.
"""

from config import Config


def get_llm(temperature: float = 0.0):
    """
    Return a LangChain-compatible chat model based on Config.LLM_PROVIDER.

    Supported providers:
        "openai"    → ChatOpenAI (requires OPENAI_API_KEY)
        "anthropic" → ChatAnthropic (requires ANTHROPIC_API_KEY)
        "lmstudio"  → ChatOpenAI pointed at local LM Studio server
    """
    provider = Config.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=Config.OPENAI_CHAT_MODEL,
            temperature=temperature,
            api_key=Config.openai_api_key() or None,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=Config.ANTHROPIC_CHAT_MODEL,
            temperature=temperature,
            api_key=Config.anthropic_api_key() or None,
        )

    elif provider == "lmstudio":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=Config.LM_STUDIO_BASE_URL,
            api_key="lm-studio",           # LM Studio ignores the key
            model=Config.LM_STUDIO_MODEL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Choose 'openai', 'anthropic', or 'lmstudio'."
        )


def get_embeddings():
    """
    Return a LangChain-compatible embeddings model based on Config.EMBEDDING_PROVIDER.

    Supported providers:
        "openai" → OpenAIEmbeddings (requires OPENAI_API_KEY)
        "local"  → HuggingFaceEmbeddings (BAAI/bge-large-en-v1.5 by default)
    """
    provider = Config.EMBEDDING_PROVIDER

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=Config.OPENAI_EMBEDDING_MODEL,
            api_key=Config.openai_api_key() or None,
        )

    elif provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. "
            "Choose 'openai' or 'local'."
        )
