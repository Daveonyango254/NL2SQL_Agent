"""
Schema Extractor Agent
Extracts combined schema using persistent embeddings and RAG
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma, FAISS


class SchemaExtractor:
    """Extract schema using persistent embeddings for RAG"""

    def __init__(self, db_id: str, db_path: Optional[Path] = None,
                 csv_paths: List[Path] = None, examples: List[Dict] = None,
                 embeddings_dir: Path = None, embeddings_model = None,
                 config: Dict = None):
        """
        Initialize Schema Extractor

        Args:
            db_id: Database identifier
            db_path: Path to SQLite database
            csv_paths: List of paths to CSV description files
            examples: List of example queries
            embeddings_dir: Directory for persistent embeddings
            embeddings_model: Embeddings model instance
            config: Configuration dictionary
        """
        self.db_id = db_id
        self.db_path = db_path
        self.csv_paths = csv_paths or []
        self.examples = examples or []
        self.config = config or {}

        # Initialize database connection
        self.db = None
        if self.db_path and self.db_path.exists():
            try:
                self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            except Exception as e:
                if self.config.get('features', {}).get('enable_debug_output'):
                    print(f"Warning: Could not connect to database: {e}")

        # Load JSON schema (exact table/column names, relationships)
        self.json_schema = None
        self.load_json_schema()

        # Load persistent RAG embeddings (for evidence/descriptions)
        self.vectorstore = None
        self.retriever = None
        if embeddings_dir and embeddings_model:
            self.load_persistent_embeddings(embeddings_dir, embeddings_model)

    def load_json_schema(self):
        """Load schema from dev_tables.json for exact table/column names"""
        # Find dev_tables.json
        json_path = Path(__file__).parent.parent.parent.parent / "data" / "bird" / "dev_tables.json"

        if not json_path.exists():
            if self.config.get('features', {}).get('enable_debug_output'):
                print(f"[WARN] dev_tables.json not found at {json_path}")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_schemas = json.load(f)

            # Find schema for this database
            for db_schema in all_schemas:
                if db_schema.get('db_id') == self.db_id:
                    self.json_schema = self._parse_json_schema(db_schema)
                    if self.config.get('features', {}).get('enable_debug_output'):
                        print(f"[OK] Loaded JSON schema for {self.db_id}")
                    return

            if self.config.get('features', {}).get('enable_debug_output'):
                print(f"[WARN] No schema found for {self.db_id} in dev_tables.json")

        except Exception as e:
            if self.config.get('features', {}).get('enable_debug_output'):
                print(f"[ERROR] Could not load JSON schema: {e}")

    def _parse_json_schema(self, db_schema: Dict) -> Dict:
        """Parse JSON schema into structured format"""
        tables = {}
        table_names = db_schema.get('table_names_original', [])
        column_info = db_schema.get('column_names_original', [])
        column_descriptions = db_schema.get('column_names', [])
        column_types = db_schema.get('column_types', [])
        foreign_keys = db_schema.get('foreign_keys', [])
        primary_keys = db_schema.get('primary_keys', [])

        # Build table structure
        for idx, table_name in enumerate(table_names):
            tables[table_name] = {
                'columns': [],
                'primary_keys': [],
                'foreign_keys': []
            }

        # Add columns
        for col_idx, (table_idx, col_name) in enumerate(column_info):
            if table_idx == -1:  # Skip "*" entry
                continue

            table_name = table_names[table_idx]
            col_description = column_descriptions[col_idx][1] if col_idx < len(column_descriptions) else col_name
            col_type = column_types[col_idx] if col_idx < len(column_types) else 'text'

            # Check if column needs backticks (has spaces or special chars)
            needs_quotes = ' ' in col_name or '(' in col_name or ')' in col_name or '%' in col_name

            tables[table_name]['columns'].append({
                'name': col_name,
                'description': col_description,
                'type': col_type,
                'needs_quotes': needs_quotes,
                'column_index': col_idx
            })

        # Add primary keys
        for pk in primary_keys:
            if isinstance(pk, list):  # Composite key
                for pk_idx in pk:
                    if pk_idx < len(column_info):
                        table_idx, col_name = column_info[pk_idx]
                        if table_idx >= 0:
                            tables[table_names[table_idx]]['primary_keys'].append(col_name)
            else:  # Single key
                if pk < len(column_info):
                    table_idx, col_name = column_info[pk]
                    if table_idx >= 0:
                        tables[table_names[table_idx]]['primary_keys'].append(col_name)

        # Add foreign keys
        fk_relationships = []
        for fk in foreign_keys:
            if len(fk) == 2:
                from_col_idx, to_col_idx = fk
                if from_col_idx < len(column_info) and to_col_idx < len(column_info):
                    from_table_idx, from_col = column_info[from_col_idx]
                    to_table_idx, to_col = column_info[to_col_idx]

                    if from_table_idx >= 0 and to_table_idx >= 0:
                        from_table = table_names[from_table_idx]
                        to_table = table_names[to_table_idx]

                        fk_relationships.append({
                            'from_table': from_table,
                            'from_column': from_col,
                            'to_table': to_table,
                            'to_column': to_col
                        })

                        tables[from_table]['foreign_keys'].append({
                            'column': from_col,
                            'references_table': to_table,
                            'references_column': to_col
                        })

        return {
            'db_id': db_schema.get('db_id'),
            'tables': tables,
            'foreign_key_relationships': fk_relationships
        }

    def load_persistent_embeddings(self, embeddings_dir: Path, embeddings_model):
        """Load precomputed embeddings from persistent storage (for evidence retrieval)"""
        # Try FAISS first (NumPy 2.0 compatible)
        faiss_directory = Path(str(embeddings_dir).replace('embeddings', 'embeddings_faiss')) / self.db_id

        if faiss_directory.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(faiss_directory),
                    embeddings_model,
                    allow_dangerous_deserialization=True
                )

                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 10}
                )

                if self.config.get('features', {}).get('enable_debug_output'):
                    print(f"[OK] Loaded FAISS embeddings for {self.db_id} (evidence retrieval)")
                return
            except Exception as e:
                if self.config.get('features', {}).get('enable_debug_output'):
                    print(f"[WARN] Could not load FAISS embeddings: {e}")

        # Fallback to Chroma (legacy, may fail with NumPy 2.0)
        persist_directory = embeddings_dir / self.db_id

        if not persist_directory.exists():
            if self.config.get('features', {}).get('enable_debug_output'):
                print(f"[WARN] No precomputed embeddings found for {self.db_id}")
                print(f"   Note: RAG evidence disabled, using JSON schema only")
            return

        try:
            self.vectorstore = Chroma(
                persist_directory=str(persist_directory),
                embedding_function=embeddings_model,
                collection_name=f"{self.db_id}_collection"
            )

            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 10}
            )

            if self.config.get('features', {}).get('enable_debug_output'):
                print(f"[OK] Loaded Chroma embeddings for {self.db_id} (evidence retrieval)")

        except Exception as e:
            # NumPy 2.0 compatibility issue or other errors
            if self.config.get('features', {}).get('enable_debug_output'):
                print(f"[WARN] Could not load persistent embeddings: {e}")
                print(f"   Note: Continuing with JSON schema only (exact names still available)")
            # Don't raise - we have JSON schema as fallback

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
        """Get combined schema: JSON schema (exact names) + RAG (evidence/descriptions)"""
        schema_context = {
            "db_id": self.db_id,
            "json_schema": None,
            "json_schema_formatted": None,
            "direct_schema": None,
            "rag_context": None,
            "evidence": [],
            "examples": [],
            "query_complexity": "simple",
            "confidence": 0.0
        }

        # PRIMARY SOURCE: JSON schema with exact table/column names
        if self.json_schema:
            schema_context["json_schema"] = self.json_schema
            schema_context["json_schema_formatted"] = self._format_json_schema()

        # BACKUP: Get direct database schema
        if self.db:
            try:
                schema_context["direct_schema"] = {
                    "tables": self.db.get_usable_table_names(),
                    "table_info": self.db.get_table_info()
                }
            except Exception as e:
                if self.config.get('features', {}).get('enable_debug_output'):
                    print(f"Warning: Could not get database schema: {e}")

        # SECONDARY SOURCE: RAG for evidence/descriptions (oracle hints)
        if self.retriever and query:
            try:
                docs = self.retriever.invoke(query)

                evidence_list = []
                schema_descriptions = []

                for doc in docs:
                    content = doc.page_content
                    source = doc.metadata.get('source_file', 'unknown')

                    # Prioritize evidence: "refers to", "commonsense evidence", value descriptions
                    is_evidence = any(phrase in content.lower() for phrase in [
                        "refers to", "commonsense evidence", "value_description",
                        "calculation:", "formula:", "eligible", "="
                    ])

                    if is_evidence:
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
                schema_context["evidence"] = evidence_list[:5]  # Increased from 3 to 5

            except Exception as e:
                if self.config.get('features', {}).get('enable_debug_output'):
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

    def _format_json_schema(self) -> str:
        """Format JSON schema for SQL generator prompt"""
        if not self.json_schema:
            return ""

        formatted = []
        formatted.append("=" * 80)
        formatted.append("EXACT DATABASE SCHEMA (Use these EXACT table and column names)")
        formatted.append("=" * 80)
        formatted.append("")

        tables = self.json_schema.get('tables', {})

        for table_name, table_info in tables.items():
            formatted.append(f"TABLE: {table_name}")
            formatted.append("-" * 60)

            # List columns
            for col in table_info['columns']:
                col_name = col['name']
                col_type = col['type']
                col_desc = col['description']
                needs_quotes = col['needs_quotes']

                # Format column line
                if needs_quotes:
                    col_display = f"`{col_name}`"
                else:
                    col_display = col_name

                # Add PRIMARY KEY indicator
                if col_name in table_info['primary_keys']:
                    formatted.append(f"  • {col_display} ({col_type}) [PRIMARY KEY] - {col_desc}")
                else:
                    formatted.append(f"  • {col_display} ({col_type}) - {col_desc}")

            # List foreign keys
            if table_info['foreign_keys']:
                formatted.append("")
                formatted.append("  Foreign Keys:")
                for fk in table_info['foreign_keys']:
                    fk_col = fk['column']
                    ref_table = fk['references_table']
                    ref_col = fk['references_column']

                    # Add backticks if needed
                    if ' ' in fk_col or '(' in fk_col:
                        fk_col = f"`{fk_col}`"
                    if ' ' in ref_col or '(' in ref_col:
                        ref_col = f"`{ref_col}`"

                    formatted.append(f"    - {fk_col} -> {ref_table}.{ref_col}")

            formatted.append("")

        # Add foreign key relationships summary
        if self.json_schema.get('foreign_key_relationships'):
            formatted.append("=" * 80)
            formatted.append("TABLE RELATIONSHIPS (for JOINs)")
            formatted.append("=" * 80)
            for fk in self.json_schema['foreign_key_relationships']:
                from_table = fk['from_table']
                from_col = fk['from_column']
                to_table = fk['to_table']
                to_col = fk['to_column']

                # Add backticks if needed
                if ' ' in from_col or '(' in from_col:
                    from_col = f"`{from_col}`"
                if ' ' in to_col or '(' in to_col:
                    to_col = f"`{to_col}`"

                formatted.append(f"  {from_table}.{from_col} -> {to_table}.{to_col}")
            formatted.append("")

        formatted.append("=" * 80)
        formatted.append("CRITICAL RULES:")
        formatted.append("1. Use EXACT table and column names from above (case-sensitive)")
        formatted.append("2. NEVER invent or guess table/column names")
        formatted.append("3. Wrap columns with backticks when they contain spaces or special characters")
        formatted.append("4. Use foreign key relationships for JOINs")
        formatted.append("=" * 80)

        return "\n".join(formatted)

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
