"""
SQL Agent with Configurable LLM Backend
Supports both OpenAI and Ollama models via config.yaml
Includes query complexity analysis, execution planning, and evidence mapping
"""

import os
import yaml
import sqlite3
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Import prompt templates
from prompt_templates import (
    get_decomposer_prompt,
    get_sql_generator_prompt,
    get_formatter_prompt,
    format_sql_only,
    format_sql_with_results,
    format_nlp_explanation,
    format_error
)
from prompt_templates.formatter import FORMATTER_PROMPT

# Import SQL extraction utility
from utils.sql_extractor import extract_sql, clean_sql
from utils.sql_validator import validate_sql_query, ValidationLevel, get_validation_summary

# Import logging utilities from evaluation module
try:
    from evaluation.logger import (
        get_eval_logger,
        get_source_logger,
        log_query_source,
        log_query_evaluation
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    get_eval_logger = None
    get_source_logger = None

# Try to import Ollama - fallback gracefully if not available
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False
        OllamaLLM = None

# Try to import HuggingFace embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceEmbeddings = None

load_dotenv()

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

# Base directory (where this script is located)
BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"

# Default configuration
DEFAULT_CONFIG = {
    "primary_model_type": "openai",
    "openai": {
        "sql_generator_model": "gpt-4o",
        "query_decomposer_model": "gpt-4o",
        "fallback_model": "gpt-4o",
        "temperature": 0
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "sql_generator_model": "llama3.1:8b",
        "query_decomposer_model": "mistral:7b",
        "fallback_to_openai": True,
        "temperature": 0
    },
    "retry": {
        "max_retries": 3,
        "fallback_after_retry": 1
    },
    "features": {
        "enable_debug_output": False,
        "log_queries": False,
        "log_file": "sql_agent_logs.txt"
    }
}


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file with defaults"""
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config:
                # Deep merge yaml_config into config
                for key, value in yaml_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value

            print(f"[OK] Loaded configuration from {config_path}")
        except Exception as e:
            print(f"[WARN] Could not load config.yaml: {e}. Using defaults.")
    else:
        print(f"[WARN] No config.yaml found at {config_path}. Using defaults.")

    return config


# Load and export CONFIG
CONFIG = load_config()

# =============================================================================
# PATH CONFIGURATION (Relative to BASE_DIR)
# =============================================================================

BIRD_DB_PATH = BASE_DIR / "data" / "bird" / "dev_databases"
BIRD_EXAMPLES_PATH = BASE_DIR / "data" / "bird" / "dev_2_examples.json"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# =============================================================================
# DATABASE UTILITIES
# =============================================================================


def discover_databases(base_path: Path = BIRD_DB_PATH) -> Dict[str, Path]:
    """Discover all available databases in BIRD dataset structure"""
    databases = {}
    if base_path.exists():
        for db_dir in base_path.iterdir():
            if db_dir.is_dir():
                sqlite_file = db_dir / f"{db_dir.name}.sqlite"
                if sqlite_file.exists():
                    databases[db_dir.name] = sqlite_file
    return databases


def get_database_path(db_id: str) -> Optional[Path]:
    """Get database path for given db_id"""
    db_path = BIRD_DB_PATH / db_id / f"{db_id}.sqlite"
    return db_path if db_path.exists() else None


def get_database_csv_paths(db_id: str) -> List[Path]:
    """Get all CSV file paths in the database_description directory"""
    csv_dir = BIRD_DB_PATH / db_id / "database_description"
    if csv_dir.exists():
        return list(csv_dir.glob("*.csv"))
    return []


def load_database_examples(db_id: str, examples_path: Path = BIRD_EXAMPLES_PATH) -> List[Dict]:
    """Load example queries for few-shot learning"""
    try:
        if examples_path.exists():
            with open(examples_path, 'r', encoding='utf-8') as f:
                all_examples = json.load(f)

            db_examples = [
                ex for ex in all_examples if ex.get('db_id') == db_id]

            # Sort by difficulty
            difficulty_order = {"simple": 0, "moderate": 1, "challenging": 2}
            db_examples.sort(key=lambda x: difficulty_order.get(
                x.get('difficulty', 'moderate'), 1))

            return db_examples
    except Exception as e:
        if CONFIG['features'].get('enable_debug_output'):
            print(f"Warning: Could not load examples: {e}")

    return []


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Track Ollama connection status globally to avoid repeated checks
_ollama_connection_verified = False
_ollama_connection_failed = False


def verify_ollama_connection():
    """Verify Ollama server is reachable. Exits with error if not."""
    global _ollama_connection_verified, _ollama_connection_failed

    # Skip if already verified or failed
    if _ollama_connection_verified:
        return True
    if _ollama_connection_failed:
        return False

    base_url = CONFIG['ollama'].get('base_url', 'http://localhost:11434')

    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=10)

        if response.status_code == 200:
            _ollama_connection_verified = True
            if CONFIG['features'].get('enable_debug_output'):
                print(f"[OK] Ollama server connected: {base_url}")
            return True
        else:
            print(f"\n[FATAL ERROR] Ollama server returned status {response.status_code}")
            print(f"  URL: {base_url}")
            print(f"  Please check your Ollama server is running.")
            _ollama_connection_failed = True
            raise SystemExit(1)

    except requests.exceptions.ConnectionError:
        print(f"\n[FATAL ERROR] Cannot connect to Ollama server")
        print(f"  URL: {base_url}")
        print(f"  Error: Connection refused - server unreachable")
        print(f"\n  Solutions:")
        print(f"    1. Start Ollama locally: 'ollama serve'")
        print(f"    2. Check if ngrok URL is correct in config.yaml")
        print(f"    3. Set primary_model_type: 'openai' in config.yaml to use OpenAI instead")
        _ollama_connection_failed = True
        raise SystemExit(1)

    except requests.exceptions.Timeout:
        print(f"\n[FATAL ERROR] Ollama server timeout")
        print(f"  URL: {base_url}")
        print(f"  Server is not responding within 10 seconds")
        _ollama_connection_failed = True
        raise SystemExit(1)

    except Exception as e:
        print(f"\n[FATAL ERROR] Ollama connection error: {e}")
        print(f"  URL: {base_url}")
        _ollama_connection_failed = True
        raise SystemExit(1)


def create_llm(model_type: str = "sql_generator", use_fallback: bool = False):
    """
    Create LLM instance based on configuration.
    Centralized factory function for all model creation.

    Args:
        model_type: "sql_generator", "query_decomposer", or "formatter"
        use_fallback: If True, use OpenAI fallback model

    Returns:
        LLM instance (ChatOpenAI or ChatOllama)
    """
    primary_type = CONFIG.get('primary_model_type', 'openai')

    # Use fallback model (always OpenAI)
    if use_fallback:
        model_name = CONFIG['openai']['fallback_model']
        temperature = CONFIG['openai'].get('temperature', 0)
        if CONFIG['features'].get('enable_debug_output'):
            print(f"  Using fallback model: OpenAI {model_name}")
        return ChatOpenAI(model=model_name, temperature=temperature)

    # Use Ollama as primary
    if primary_type == 'ollama':
        # Verify connection (will exit if failed)
        verify_ollama_connection()

        # Import ChatOllama for chat-based interface
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        base_url = CONFIG['ollama'].get('base_url', 'http://localhost:11434')
        # Dynamic model lookup based on model_type
        model_name = CONFIG['ollama'].get(f'{model_type}_model', 'llama3.1:8b')
        temperature = CONFIG['ollama'].get('temperature', 0)

        if CONFIG['features'].get('enable_debug_output'):
            print(f"  Using Ollama model: {model_name}")

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )

    # Use OpenAI as primary
    model_name = CONFIG['openai'].get(f'{model_type}_model', 'gpt-4o')
    temperature = CONFIG['openai'].get('temperature', 0)

    if CONFIG['features'].get('enable_debug_output'):
        print(f"  Using OpenAI model: {model_name}")

    return ChatOpenAI(model=model_name, temperature=temperature)


class LLMConfig:
    """Configurable LLM factory based on config.yaml settings"""

    @staticmethod
    def get_decomposer_llm(regenerate_count: int = 0):
        """Get LLM for query decomposition"""
        fallback_after = CONFIG['retry'].get('fallback_after_retry', 1)
        use_fallback = regenerate_count >= fallback_after
        return create_llm(model_type="query_decomposer", use_fallback=use_fallback)

    @staticmethod
    def get_sql_generator_llm(regenerate_count: int = 0):
        """Get LLM for SQL generation"""
        fallback_after = CONFIG['retry'].get('fallback_after_retry', 1)
        use_fallback = regenerate_count >= fallback_after
        return create_llm(model_type="sql_generator", use_fallback=use_fallback)

    @staticmethod
    def get_formatter_llm():
        """Get LLM for result formatting"""
        return create_llm(model_type="sql_generator", use_fallback=False)

    @staticmethod
    def get_embeddings():
        """Get embeddings model - prefer local HuggingFace for speed"""
        if HUGGINGFACE_AVAILABLE:
            try:
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                if CONFIG['features'].get('enable_debug_output'):
                    print(f"HuggingFace embeddings failed: {e}. Using OpenAI.")

        return OpenAIEmbeddings()


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SQLAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    decomposed_queries: List[str]  # Legacy - kept for compatibility
    execution_plan: List[str]  # Step-by-step execution plan
    evidence_mapping: List[str]  # NL → DB element mappings
    schema_context: Dict[str, Any]
    relevant_tables: List[str]
    sql_query: str
    sql_results: Any
    formatted_response: str
    error_count: int
    validation_status: bool
    output_mode: Literal["sql_only", "sql_with_results", "nlp_explanation"]
    confidence_score: float
    db_id: str
    db_path: Optional[str]
    regenerate_count: int  # Track regeneration attempts


# =============================================================================
# SCHEMA EXTRACTOR (with RAG support)
# =============================================================================

class SchemaExtractor:
    """Extract schema using persistent embeddings for RAG"""

    def __init__(self, db_id: str):
        self.db_id = db_id
        self.db_path = get_database_path(db_id)
        self.csv_paths = get_database_csv_paths(db_id)
        self.examples = load_database_examples(db_id)

        # Initialize database connection
        self.db = None
        if self.db_path and self.db_path.exists():
            try:
                self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            except Exception as e:
                if CONFIG['features'].get('enable_debug_output'):
                    print(f"Warning: Could not connect to database: {e}")

        # Load persistent RAG embeddings
        self.vectorstore = None
        self.retriever = None
        self.load_persistent_embeddings()

    def load_persistent_embeddings(self):
        """Load precomputed embeddings from persistent storage"""
        persist_directory = EMBEDDINGS_DIR / self.db_id

        if not persist_directory.exists():
            if CONFIG['features'].get('enable_debug_output'):
                print(f"[WARN] No precomputed embeddings found for {self.db_id}")
                print(
                    f"   Run: python embed_training_data.py --db {self.db_id}")
            return

        try:
            embeddings = LLMConfig.get_embeddings()

            self.vectorstore = Chroma(
                persist_directory=str(persist_directory),
                embedding_function=embeddings,
                collection_name=f"{self.db_id}_collection"
            )

            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 10}
            )

            if CONFIG['features'].get('enable_debug_output'):
                print(f"[OK] Loaded persistent embeddings for {self.db_id}")

        except Exception as e:
            if CONFIG['features'].get('enable_debug_output'):
                print(f"Warning: Could not load persistent embeddings: {e}")

    def get_similar_examples(self, query: str, limit: int = 3) -> List[Dict]:
        """Find similar example queries for few-shot learning"""
        if not self.examples:
            return []

        query_lower = query.lower()
        scored_examples = []

        for example in self.examples:
            example_question = example.get('question', '').lower()
            common_words = set(query_lower.split()) & set(
                example_question.split())
            score = len(common_words)

            if score > 0:
                scored_examples.append((score, example))

        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex[1] for ex in scored_examples[:limit]]

    def get_combined_schema(self, query: str) -> Dict[str, Any]:
        """Get combined schema from database and RAG sources"""
        schema_context = {
            "db_id": self.db_id,
            "direct_schema": None,
            "rag_context": None,
            "evidence": [],
            "examples": [],
            "query_complexity": "simple",
            "confidence": 0.0
        }

        # Get direct database schema
        if self.db:
            try:
                schema_context["direct_schema"] = {
                    "tables": self.db.get_usable_table_names(),
                    "table_info": self.db.get_table_info()
                }
            except Exception as e:
                if CONFIG['features'].get('enable_debug_output'):
                    print(f"Warning: Could not get database schema: {e}")

        # Get RAG context from persistent embeddings
        if self.retriever and query:
            try:
                docs = self.retriever.invoke(query)

                evidence_list = []
                schema_descriptions = []

                for doc in docs:
                    content = doc.page_content
                    source = doc.metadata.get('source_file', 'unknown')

                    if "refers to" in content.lower() or "=" in content:
                        evidence_list.append({
                            "source": source,
                            "content": content
                        })

                    schema_descriptions.append({
                        "source": source,
                        "description": content
                    })

                schema_context["rag_context"] = {
                    "relevant_descriptions": schema_descriptions[:5],
                    "metadata": [doc.metadata for doc in docs[:5]]
                }
                schema_context["evidence"] = evidence_list[:3]

            except Exception as e:
                if CONFIG['features'].get('enable_debug_output'):
                    print(f"Warning: Could not get RAG context: {e}")

        # Add similar examples
        similar_examples = self.get_similar_examples(query)
        if similar_examples:
            schema_context["examples"] = similar_examples

            for example in similar_examples:
                if example.get('evidence'):
                    schema_context["evidence"].append({
                        "source": "BIRD examples",
                        "content": example['evidence']
                    })

        # Analyze complexity
        schema_context["query_complexity"] = self._analyze_query_complexity(
            query, schema_context)
        schema_context["confidence"] = self._calculate_relevance_confidence(
            query, schema_context)

        return schema_context

    def _analyze_query_complexity(self, query: str, schema: Dict) -> str:
        """Analyze query complexity"""
        query_lower = query.lower()

        simple_indicators = [
            query_lower.count(' ') < 8,
            'top' in query_lower and any(
                str(i) in query_lower for i in range(1, 11)),
            'list' in query_lower or 'show' in query_lower,
            query_lower.count('and') == 0 and query_lower.count('or') == 0,
            len(schema.get("examples", [])) > 0 and schema["examples"][0].get(
                'difficulty') == 'simple'
        ]

        complex_indicators = [
            query_lower.count(' ') > 15,
            query_lower.count('and') + query_lower.count('or') >= 2,
            'average' in query_lower or 'sum' in query_lower,
            'compare' in query_lower or 'difference' in query_lower,
            'for each' in query_lower or 'per' in query_lower,
            len(schema.get("examples", [])) > 0 and schema["examples"][0].get(
                'difficulty') in ['moderate', 'challenging']
        ]

        simple_count = sum(simple_indicators)
        complex_count = sum(complex_indicators)

        if simple_count >= 3 or (simple_count > complex_count and len(schema.get("examples", [])) > 0):
            return "simple"
        elif complex_count >= 3:
            return "complex"
        else:
            return "moderate"

    def _calculate_relevance_confidence(self, query: str, schema: Dict) -> float:
        """Calculate confidence score"""
        confidence = 0.0

        if schema.get("examples"):
            confidence += 0.4
        if len(schema.get("evidence", [])) >= 2:
            confidence += 0.3
        if schema.get("rag_context"):
            confidence += 0.2
        if schema.get("direct_schema"):
            confidence += 0.1

        return min(confidence, 1.0)


# =============================================================================
# AGENT CLASSES
# =============================================================================

class QueryDecomposer:
    """Decompose complex queries and generate evidence mapping"""

    def __init__(self, regenerate_count: int = 0):
        self.llm = LLMConfig.get_decomposer_llm(regenerate_count)
        self.decompose_prompt = get_decomposer_prompt()

    def decompose(self, query: str, schema: Dict) -> Dict[str, Any]:
        """Decompose query and generate evidence mapping"""
        chain = self.decompose_prompt | self.llm

        response = chain.invoke({
            "query": query,
            "direct_schema": str(schema.get("direct_schema", {})),
            "rag_context": str(schema.get("rag_context", {}))
        })

        content = response.content if hasattr(
            response, 'content') else str(response)
        return self._parse_decomposition_response(content)

    def _parse_decomposition_response(self, content: str) -> Dict[str, Any]:
        """Parse the decomposition response into structured format"""
        result = {"execution_plan": [], "evidence_mapping": []}

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if 'EXECUTION PLAN' in line.upper():
                current_section = 'execution_plan'
                continue
            elif 'EVIDENCE MAPPING' in line.upper() or 'EVIDENCE' in line.upper():
                current_section = 'evidence_mapping'
                continue

            if current_section == 'execution_plan':
                cleaned = line.lstrip('0123456789.-) ')
                if cleaned:
                    result["execution_plan"].append(cleaned)
            elif current_section == 'evidence_mapping':
                if '→' in line or '->' in line or ':' in line:
                    result["evidence_mapping"].append(line.lstrip('- '))

        return result


class SQLGenerator:
    """Generate SQL using configured LLM"""

    def __init__(self, regenerate_count: int = 0):
        self.regenerate_count = regenerate_count
        self.llm = LLMConfig.get_sql_generator_llm(regenerate_count)
        self.sql_prompt = get_sql_generator_prompt()

    def format_examples(self, examples: List[Dict]) -> str:
        """Format examples for the prompt"""
        if not examples:
            return "No similar examples available."

        formatted = []
        for i, ex in enumerate(examples[:3], 1):
            formatted.append(f"""
Example {i}:
Question: {ex.get('question', 'N/A')}
Evidence: {ex.get('evidence', 'N/A')}
SQL: {ex.get('SQL', 'N/A')}
""")
        return "\n".join(formatted)

    def generate(self, query: str, schema: Dict, execution_plan: List[str], evidence_mapping: List[str]) -> str:
        """Generate SQL query"""
        chain = self.sql_prompt | self.llm

        examples_str = self.format_examples(schema.get("examples", []))
        plan_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(
            execution_plan)]) if execution_plan else "Direct SQL generation"
        evidence_str = "\n".join(
            evidence_mapping) if evidence_mapping else "No evidence mapping"

        response = chain.invoke({
            "query": query,
            "direct_schema": str(schema.get("direct_schema", {})),
            "rag_context": str(schema.get("rag_context", {})),
            "evidence": str(schema.get("evidence", [])),
            "examples": examples_str,
            "execution_plan": plan_str,
            "evidence_mapping": evidence_str
        })

        content = response.content if hasattr(
            response, 'content') else str(response)
        # Use robust SQL extraction from utils
        return extract_sql(content)


# =============================================================================
# AGENT NODES
# =============================================================================

def schema_extraction_node(state: SQLAgentState) -> SQLAgentState:
    """Extract combined schema using persistent embeddings"""
    db_id = state.get("db_id")

    if not db_id:
        state["error_count"] = state.get("error_count", 0) + 1
        state["messages"].append(AIMessage(content="No database ID provided"))
        state["schema_context"] = {}
        return state

    extractor = SchemaExtractor(db_id=db_id)
    schema = extractor.get_combined_schema(state["user_query"])
    state["schema_context"] = schema

    if extractor.db_path:
        state["db_path"] = str(extractor.db_path)

    sources = []
    if schema.get("direct_schema"):
        sources.append("direct database schema")
    if schema.get("rag_context"):
        sources.append("persistent embeddings")

    msg = f"Schema extracted using: {', '.join(sources)}"
    if schema.get("evidence"):
        msg += f" with {len(schema['evidence'])} evidence items"

    state["messages"].append(AIMessage(content=msg))
    return state


def query_decomposer_node(state: SQLAgentState) -> SQLAgentState:
    """Decompose complex query (conditional execution)"""
    schema_context = state.get("schema_context", {})
    query_complexity = schema_context.get("query_complexity", "moderate")
    confidence = schema_context.get("confidence", 0.0)
    regenerate_count = state.get("regenerate_count", 0)

    # Skip decomposition for simple queries with high confidence (first attempt only)
    skip_decomposition = (
        query_complexity == "simple" and
        confidence >= 0.6 and
        state.get("error_count", 0) == 0 and
        regenerate_count == 0
    )

    if skip_decomposition:
        state["execution_plan"] = []
        state["evidence_mapping"] = []
        state["decomposed_queries"] = []
        state["messages"].append(
            AIMessage(
                content=f"Skipping decomposition (simple query, confidence: {confidence:.2f})")
        )
    else:
        decomposer = QueryDecomposer(regenerate_count=regenerate_count)
        result = decomposer.decompose(state["user_query"], schema_context)

        state["execution_plan"] = result.get("execution_plan", [])
        state["evidence_mapping"] = result.get("evidence_mapping", [])
        state["decomposed_queries"] = []

        state["messages"].append(
            AIMessage(
                content=f"Query decomposed: {len(state['execution_plan'])} steps, {len(state['evidence_mapping'])} evidence mappings")
        )

    return state


def sql_generator_node(state: SQLAgentState) -> SQLAgentState:
    """Generate SQL using configured LLM"""
    regenerate_count = state.get("regenerate_count", 0)

    # Increment regenerate count if this is a retry
    if state.get("validation_status") is False:
        regenerate_count += 1
        state["regenerate_count"] = regenerate_count

    generator = SQLGenerator(regenerate_count=regenerate_count)
    sql = generator.generate(
        state["user_query"],
        state.get("schema_context", {}),
        state.get("execution_plan", []),
        state.get("evidence_mapping", [])
    )

    state["sql_query"] = sql

    # Determine which model was used
    primary_type = CONFIG.get('primary_model_type', 'openai')
    fallback_after = CONFIG['retry'].get('fallback_after_retry', 1)

    if primary_type == 'ollama' and regenerate_count < fallback_after:
        model_info = f"Ollama ({CONFIG['ollama'].get('sql_generator_model', 'local')})"
    else:
        model_info = f"OpenAI ({CONFIG['openai'].get('sql_generator_model', 'gpt-4o')})"

    state["messages"].append(
        AIMessage(content=f"SQL generated using {model_info}"))

    # Calculate confidence
    schema_context = state.get("schema_context", {})
    base_confidence = 0.5

    if schema_context.get("rag_context"):
        base_confidence += 0.2
    if state.get("evidence_mapping"):
        base_confidence += 0.15
    if state.get("execution_plan"):
        base_confidence += 0.15
    if regenerate_count > 0:
        base_confidence = min(0.95, base_confidence + 0.1)

    complexity = len(state.get("execution_plan", []))
    state["confidence_score"] = max(0.3, base_confidence - (complexity * 0.03))

    return state


def executor_validator_node(state: SQLAgentState) -> SQLAgentState:
    """Execute SQL and validate results with enhanced multi-layer validation"""
    db_path = state.get("db_path")
    sql_query = state.get("sql_query", "")
    user_query = state.get("user_query", "")
    schema_context = state.get("schema_context", {})

    if not db_path:
        state["sql_results"] = "Query generated but not executed (no database connection)"
        state["validation_status"] = True
        return state

    # Step 1: Pre-execution validation (structural + semantic)
    is_valid_pre, pre_results, confidence_adj = validate_sql_query(
        sql=sql_query,
        question=user_query,
        db_path=db_path,
        schema_context=schema_context,
        results=None  # No results yet
    )

    # Check for critical errors before execution
    pre_errors = [r for r in pre_results if r.level == ValidationLevel.ERROR]
    if pre_errors:
        state["error_count"] = state.get("error_count", 0) + 1
        state["validation_status"] = False
        error_msgs = "; ".join([r.message for r in pre_errors])
        state["messages"].append(
            AIMessage(content=f"Pre-execution validation failed: {error_msgs}")
        )
        return state

    # Step 2: Execute the query
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(sql_query)
        results = cursor.fetchall()
        conn.close()

        state["sql_results"] = results

        # Step 3: Post-execution validation (result checks)
        is_valid_post, post_results, post_confidence_adj = validate_sql_query(
            sql=sql_query,
            question=user_query,
            db_path=db_path,
            schema_context=schema_context,
            results=results
        )

        # Combine confidence adjustments
        total_confidence_adj = confidence_adj + post_confidence_adj
        state["confidence_score"] = max(0.1, state.get("confidence_score", 0.5) + total_confidence_adj)

        # Check for post-execution errors
        post_errors = [r for r in post_results if r.level == ValidationLevel.ERROR]
        post_warnings = [r for r in post_results if r.level == ValidationLevel.WARNING]

        if post_errors:
            # Critical errors - mark as failed for regeneration
            state["error_count"] = state.get("error_count", 0) + 1
            state["validation_status"] = False
            error_msgs = "; ".join([r.message for r in post_errors])
            state["messages"].append(
                AIMessage(content=f"Post-execution validation failed: {error_msgs}")
            )
        elif len(post_warnings) >= 3:
            # Too many warnings - likely incorrect, regenerate
            state["error_count"] = state.get("error_count", 0) + 1
            state["validation_status"] = False
            warning_msgs = "; ".join([r.message for r in post_warnings[:3]])
            state["messages"].append(
                AIMessage(content=f"Multiple validation warnings (likely incorrect): {warning_msgs}")
            )
        else:
            # Passed validation
            state["validation_status"] = True
            validation_summary = get_validation_summary(post_results)
            state["messages"].append(
                AIMessage(content=f"Query executed: {len(results)} rows. Validation: {validation_summary}")
            )

            # Log warnings if any
            if post_warnings and CONFIG['features'].get('enable_debug_output'):
                for warning in post_warnings:
                    print(f"  [VALIDATION WARNING] {warning.message}")

    except Exception as e:
        state["error_count"] = state.get("error_count", 0) + 1
        state["validation_status"] = False
        state["messages"].append(
            AIMessage(content=f"Execution error: {str(e)}")
        )

    return state


def formatter_node(state: SQLAgentState) -> SQLAgentState:
    """Format results based on output mode"""
    output_mode = state.get("output_mode", "nlp_explanation")

    if output_mode == "sql_only":
        state["formatted_response"] = format_sql_only(state['sql_query'])

    elif output_mode == "sql_with_results":
        state["formatted_response"] = format_sql_with_results(
            state['sql_query'],
            str(state['sql_results'])
        )

    else:  # nlp_explanation
        llm = LLMConfig.get_formatter_llm()
        formatter_prompt = get_formatter_prompt()

        chain = formatter_prompt | llm
        response = chain.invoke({
            "question": state["user_query"],
            "sql": state["sql_query"],
            "results": str(state["sql_results"]),
            "schema_context": str(state.get("schema_context", {}).get("evidence", []))
        })

        content = response.content if hasattr(
            response, 'content') else str(response)

        state["formatted_response"] = format_nlp_explanation(
            state['user_query'],
            state['sql_query'],
            content,
            state['confidence_score']
        )

    return state


def error_handler_node(state: SQLAgentState) -> SQLAgentState:
    """Handle errors and prepare error message"""
    state["formatted_response"] = format_error(
        state['user_query'],
        state.get('sql_query', 'Not generated'),
        state.get('error_count', 0)
    )
    return state


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def should_retry(state: SQLAgentState) -> str:
    """Determine next step based on validation"""
    max_retries = CONFIG['retry'].get('max_retries', 3)

    if state.get("validation_status", False):
        return "format"
    elif state.get("error_count", 0) >= max_retries:
        return "format_error"
    else:
        # Check if simple query failed - might need decomposition
        schema_context = state.get("schema_context", {})
        query_complexity = schema_context.get("query_complexity", "moderate")
        execution_plan = state.get("execution_plan", [])

        if query_complexity == "simple" and len(execution_plan) == 0:
            return "re_decompose"
        else:
            return "regenerate"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_sql_agent_graph():
    """Build the SQL agent graph"""
    graph = StateGraph(SQLAgentState)

    # Add nodes
    graph.add_node("Schema_Extraction", schema_extraction_node)
    graph.add_node("Query_Decomposer", query_decomposer_node)
    graph.add_node("SQL_Generator", sql_generator_node)
    graph.add_node("Executor_Validator", executor_validator_node)
    graph.add_node("Formatter", formatter_node)
    graph.add_node("Error_Handler", error_handler_node)

    # Linear workflow
    graph.add_edge(START, "Schema_Extraction")
    graph.add_edge("Schema_Extraction", "Query_Decomposer")
    graph.add_edge("Query_Decomposer", "SQL_Generator")
    graph.add_edge("SQL_Generator", "Executor_Validator")

    # Conditional routing
    graph.add_conditional_edges(
        "Executor_Validator",
        should_retry,
        {
            "format": "Formatter",
            "regenerate": "SQL_Generator",
            "re_decompose": "Query_Decomposer",
            "format_error": "Error_Handler"
        }
    )

    graph.add_edge("Formatter", END)
    graph.add_edge("Error_Handler", END)

    return graph.compile()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_sql_agent(
    query: str,
    db_id: str,
    output_mode: str = "nlp_explanation"
) -> str:
    """
    Run the SQL agent with specified configuration

    Args:
        query: User's natural language query
        db_id: Database identifier (required)
        output_mode: One of ["sql_only", "sql_with_results", "nlp_explanation"]

    Returns:
        Formatted response string
    """
    if not db_id:
        raise ValueError("db_id is required to identify the database")

    # Build the graph
    agent = build_sql_agent_graph()

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "db_id": db_id,
        "user_query": query,
        "output_mode": output_mode,
        "error_count": 0,
        "regenerate_count": 0,
        "confidence_score": 0.0,
        "execution_plan": [],
        "evidence_mapping": [],
        "decomposed_queries": []
    }

    # Run the agent
    result = agent.invoke(initial_state)

    # To Enable Logging ##*****************************************************
    # Determine which model was used for logging
    regenerate_count = result.get("regenerate_count", 0)
    primary_type = CONFIG.get('primary_model_type', 'openai')
    fallback_after = CONFIG['retry'].get('fallback_after_retry', 1)
    is_fallback = regenerate_count >= fallback_after

    if primary_type == 'ollama' and not is_fallback:
        model_provider = "ollama"
        model_name = CONFIG['ollama'].get('sql_generator_model', 'local')
    else:
        model_provider = "openai"
        model_name = CONFIG['openai'].get(
            'fallback_model' if is_fallback else 'sql_generator_model', 'gpt-4o')

    # Log query source tracking
    if LOGGING_AVAILABLE:
        source_logger = get_source_logger()
        eval_logger = get_eval_logger()

        # Determine success based on validation status
        success = result.get("validation_status", False)
        sql_query = result.get("sql_query", "")
        rows_returned = len(result.get("sql_results", [])) if isinstance(
            result.get("sql_results"), list) else None

        # Log source tracking
        log_query_source(
            source_logger=source_logger,
            db_id=db_id,
            query=query,
            model_provider=model_provider,
            model_name=model_name,
            regenerate_count=regenerate_count,
            is_fallback=is_fallback,
            success=success,
            sql_query=sql_query
        )

        # Log evaluation results
        error_msg = None
        if not success:
            # Extract error from messages
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'content') and 'error' in msg.content.lower():
                    error_msg = msg.content
                    break

        log_query_evaluation(
            eval_logger=eval_logger,
            db_id=db_id,
            query=query,
            sql_query=sql_query,
            success=success,
            error_message=error_msg,
            rows_returned=rows_returned,
            confidence_score=result.get("confidence_score")
        )

    # Legacy file logging if enabled
    if CONFIG['features'].get('log_queries'):
        log_file = BASE_DIR / \
            CONFIG['features'].get('log_file', 'sql_agent_logs.txt')
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Query: {query}\n")
                f.write(f"DB: {db_id}\n")
                f.write(f"SQL: {result.get('sql_query', 'N/A')}\n")
                f.write(f"Model: {model_provider}/{model_name}\n")
                f.write(f"Fallback: {is_fallback}\n")
                f.write(f"{'='*60}\n")
        except Exception:
            pass

    return result["formatted_response"]


# =============================================================================
# CLI EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python SQL_Agent.py <db_id> <question> <output_mode>")
        print("Example: python SQL_Agent.py california_schools \"List all charter schools\" sql_with_results")
        print(f"\nAvailable databases: {list(discover_databases().keys())}")
        print(f"\nUsing: {CONFIG.get('primary_model_type', 'openai')} models")
        sys.exit(1)

    db_id = sys.argv[1]
    question = sys.argv[2]
    output_mode = sys.argv[3]

    print(f"\n{'='*60}")
    print(
        f"SQL Agent - {CONFIG.get('primary_model_type', 'openai').upper()} Mode")
    print(f"Database: {db_id}")
    print(f"Question: {question}")
    print(f"Output Mode: {output_mode}")
    print(f"{'='*60}\n")

    results = run_sql_agent(
        query=question,
        db_id=db_id,
        output_mode=output_mode
    )

    print(results)
