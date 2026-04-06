"""
LLM Factory Module
Centralized LLM creation and configuration management
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Track Ollama connection status globally to avoid repeated checks
_ollama_connection_verified = False
_ollama_connection_failed = False

# Try to import Ollama - fallback gracefully if not available
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False
        ChatOllama = None

# Try to import HuggingFace embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceEmbeddings = None


def verify_ollama_connection(config: Dict) -> bool:
    """Verify Ollama server is reachable"""
    global _ollama_connection_verified, _ollama_connection_failed

    # Skip if already verified or failed
    if _ollama_connection_verified:
        return True
    if _ollama_connection_failed:
        return False

    base_url = config['ollama'].get('base_url', 'http://localhost:11434')

    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=10)

        if response.status_code == 200:
            _ollama_connection_verified = True
            if config['features'].get('enable_debug_output'):
                print(f"[OK] Ollama server connected: {base_url}")
            return True
        else:
            print(f"\n[FATAL ERROR] Ollama server returned status {response.status_code}")
            print(f"  URL: {base_url}")
            print(f"  Please check your Ollama server is running.")
            _ollama_connection_failed = True
            raise SystemExit(1)

    except Exception as e:
        print(f"\n[FATAL ERROR] Cannot connect to Ollama server")
        print(f"  URL: {base_url}")
        print(f"  Error: {e}")
        print(f"\n  Solutions:")
        print(f"    1. Start Ollama locally: 'ollama serve'")
        print(f"    2. Check if ngrok URL is correct in config.yaml")
        print(f"    3. Set primary_model_type: 'openai' in config.yaml")
        _ollama_connection_failed = True
        raise SystemExit(1)


def create_llm(model_type: str, config: Dict, use_fallback: bool = False):
    """
    Create LLM instance based on configuration

    Args:
        model_type: "sql_generator", "query_decomposer", or "formatter"
        config: Configuration dictionary
        use_fallback: If True, use OpenAI fallback model

    Returns:
        LLM instance (ChatOpenAI or ChatOllama)
    """
    primary_type = config.get('primary_model_type', 'openai')

    # Use fallback model (always OpenAI)
    if use_fallback:
        model_name = config['openai']['fallback_model']
        temperature = config['openai'].get('temperature', 0)
        if config['features'].get('enable_debug_output'):
            print(f"  Using fallback model: OpenAI {model_name}")
        return ChatOpenAI(model=model_name, temperature=temperature)

    # Use Ollama as primary
    if primary_type == 'ollama':
        # Verify connection
        verify_ollama_connection(config)

        base_url = config['ollama'].get('base_url', 'http://localhost:11434')
        model_name = config['ollama'].get(f'{model_type}_model', 'llama3.1:8b')
        temperature = config['ollama'].get('temperature', 0)

        if config['features'].get('enable_debug_output'):
            print(f"  Using Ollama model: {model_name}")

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )

    # Use OpenAI as primary
    model_name = config['openai'].get(f'{model_type}_model', 'gpt-4o')
    temperature = config['openai'].get('temperature', 0)

    if config['features'].get('enable_debug_output'):
        print(f"  Using OpenAI model: {model_name}")

    return ChatOpenAI(model=model_name, temperature=temperature)


class LLMConfig:
    """Configurable LLM factory based on config.yaml settings"""

    def __init__(self, config: Dict):
        self.config = config
        self._embeddings_cache = None  # Cache embeddings model to avoid reloading

    def get_decomposer_llm(self, regenerate_count: int = 0):
        """Get LLM for query decomposition"""
        # Check if fallback is enabled
        enable_fallback = self.config['retry'].get('enable_fallback', True)
        if not enable_fallback:
            # Fallback disabled - always use primary model
            return create_llm(model_type="query_decomposer", config=self.config, use_fallback=False)

        # Fallback enabled - check retry count
        fallback_after = self.config['retry'].get('fallback_after_retry', 1)
        use_fallback = regenerate_count >= fallback_after
        return create_llm(model_type="query_decomposer", config=self.config, use_fallback=use_fallback)

    def get_sql_generator_llm(self, regenerate_count: int = 0):
        """Get LLM for SQL generation"""
        # Check if fallback is enabled
        enable_fallback = self.config['retry'].get('enable_fallback', True)
        if not enable_fallback:
            # Fallback disabled - always use primary model
            return create_llm(model_type="sql_generator", config=self.config, use_fallback=False)

        # Fallback enabled - check retry count
        fallback_after = self.config['retry'].get('fallback_after_retry', 1)
        use_fallback = regenerate_count >= fallback_after
        return create_llm(model_type="sql_generator", config=self.config, use_fallback=use_fallback)

    def get_formatter_llm(self):
        """Get LLM for result formatting"""
        return create_llm(model_type="sql_generator", config=self.config, use_fallback=False)

    def get_embeddings(self):
        """Get embeddings model - prefer local HuggingFace for speed (cached after first load)"""
        # Return cached embeddings if already loaded
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        if HUGGINGFACE_AVAILABLE:
            try:
                # Suppress HuggingFace progress bars to avoid conflicts with tqdm
                import os
                os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

                self._embeddings_cache = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True},
                    show_progress=False
                )
                return self._embeddings_cache
            except Exception as e:
                if self.config['features'].get('enable_debug_output'):
                    print(f"HuggingFace embeddings failed: {e}. Using OpenAI.")

        self._embeddings_cache = OpenAIEmbeddings()
        return self._embeddings_cache
