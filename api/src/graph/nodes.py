"""
Node functions for SQL Agent graph workflow
These are wrapper functions that integrate agent classes into the graph
"""

from pathlib import Path
from langchain_core.messages import AIMessage
from .state import SQLAgentState
from ..agents.schema_extractor import SchemaExtractor
from ..agents.query_decomposer import QueryDecomposer
from ..agents.sql_generator import SQLGenerator
from ..agents.validator import execute_and_validate, get_validation_summary


def create_schema_extraction_node(
    get_database_path_func,
    get_csv_paths_func,
    load_examples_func,
    embeddings_dir: Path,
    get_embeddings_func,
    config: dict
):
    """Factory function to create schema extraction node with dependencies"""

    def schema_extraction_node(state: SQLAgentState) -> SQLAgentState:
        """Extract combined schema using persistent embeddings"""
        db_id = state.get("db_id")

        if not db_id:
            state["error_count"] = state.get("error_count", 0) + 1
            state["messages"].append(AIMessage(content="No database ID provided"))
            state["schema_context"] = {}
            return state

        # Get database resources
        db_path = get_database_path_func(db_id)
        csv_paths = get_csv_paths_func(db_id)
        examples = load_examples_func(db_id)
        embeddings_model = get_embeddings_func()

        # Create extractor
        extractor = SchemaExtractor(
            db_id=db_id,
            db_path=db_path,
            csv_paths=csv_paths,
            examples=examples,
            embeddings_dir=embeddings_dir,
            embeddings_model=embeddings_model,
            config=config
        )

        schema = extractor.get_combined_schema(state["user_query"])
        state["schema_context"] = schema

        if db_path:
            state["db_path"] = str(db_path)

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

    return schema_extraction_node


def create_query_decomposer_node(get_llm_func, get_prompt_func):
    """Factory function to create query decomposer node with dependencies"""

    def query_decomposer_node(state: SQLAgentState) -> SQLAgentState:
        """
        Decompose query ALWAYS - no conditional skipping
        Following strict graph flow
        """
        schema_context = state.get("schema_context", {})
        regenerate_count = state.get("regenerate_count", 0)

        # Get LLM and prompt
        llm = get_llm_func(regenerate_count)
        prompt = get_prompt_func()

        # Create decomposer and execute (ALWAYS - no skipping)
        decomposer = QueryDecomposer(llm=llm, decompose_prompt=prompt)
        result = decomposer.decompose(state["user_query"], schema_context)

        state["execution_plan"] = result.get("execution_plan", [])
        state["evidence_mapping"] = result.get("evidence_mapping", [])
        state["decomposed_queries"] = []

        state["messages"].append(
            AIMessage(
                content=f"Query decomposed: {len(state['execution_plan'])} steps, {len(state['evidence_mapping'])} evidence mappings")
        )

        return state

    return query_decomposer_node


def create_sql_generator_node(get_llm_func, get_prompt_func, extract_sql_func, config: dict):
    """Factory function to create SQL generator node with dependencies"""

    def sql_generator_node(state: SQLAgentState) -> SQLAgentState:
        """Generate SQL using configured LLM"""
        regenerate_count = state.get("regenerate_count", 0)

        # Increment regenerate count if this is a retry
        if state.get("validation_status") is False:
            regenerate_count += 1
            state["regenerate_count"] = regenerate_count

        # Get LLM and prompt
        llm = get_llm_func(regenerate_count)
        prompt = get_prompt_func()

        # Create generator
        generator = SQLGenerator(
            llm=llm,
            sql_prompt=prompt,
            sql_extractor_func=extract_sql_func
        )

        sql = generator.generate(
            state["user_query"],
            state.get("schema_context", {}),
            state.get("execution_plan", []),
            state.get("evidence_mapping", [])
        )

        state["sql_query"] = sql

        # Determine which model was used
        primary_type = config.get('primary_model_type', 'openai')
        fallback_after = config['retry'].get('fallback_after_retry', 1)

        if primary_type == 'ollama' and regenerate_count < fallback_after:
            model_info = f"Ollama ({config['ollama'].get('sql_generator_model', 'local')})"
        else:
            model_info = f"OpenAI ({config['openai'].get('sql_generator_model', 'gpt-4o')})"

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

    return sql_generator_node


def create_executor_validator_node(config: dict):
    """Factory function to create executor/validator node"""

    def executor_validator_node(state: SQLAgentState) -> SQLAgentState:
        """Execute SQL and validate results with enhanced multi-layer validation"""
        db_path = state.get("db_path")
        sql_query = state.get("sql_query", "")
        user_query = state.get("user_query", "")
        schema_context = state.get("schema_context", {})

        if not db_path:
            state["sql_results"] = "Query generated but not executed (no database connection)"
            state["validation_status"] = False  # ✅ FIX: Should fail validation if no db_path
            state["error_count"] = state.get("error_count", 0) + 1
            state["messages"].append(
                AIMessage(content="Validation failed: No database path available")
            )
            return state

        # Execute and validate
        is_valid, results, validation_results, confidence_adj = execute_and_validate(
            sql_query=sql_query,
            user_query=user_query,
            db_path=db_path,
            schema_context=schema_context,
            config=config
        )

        state["sql_results"] = results
        state["validation_status"] = is_valid

        # Adjust confidence
        state["confidence_score"] = max(0.1, state.get("confidence_score", 0.5) + confidence_adj)

        if is_valid:
            validation_summary = get_validation_summary(validation_results)
            row_count = len(results) if isinstance(results, list) else 0
            state["messages"].append(
                AIMessage(
                    content=f"Query executed: {row_count} rows. Validation: {validation_summary}")
            )
        else:
            # ✅ FIX: Increment regenerate_count when validation fails
            state["regenerate_count"] = state.get("regenerate_count", 0) + 1
            state["error_count"] = state.get("error_count", 0) + 1
            error_msgs = "; ".join([r.message for r in validation_results if r.level.value == "error"])
            state["messages"].append(
                AIMessage(content=f"Validation failed (attempt {state['regenerate_count']}): {error_msgs}")
            )

        return state

    return executor_validator_node


def create_formatter_node(get_llm_func, get_prompt_func, format_funcs: dict):
    """Factory function to create formatter node"""

    def formatter_node(state: SQLAgentState) -> SQLAgentState:
        """Format results based on output mode"""
        output_mode = state.get("output_mode", "nlp_explanation")

        if output_mode == "sql_only":
            state["formatted_response"] = format_funcs['format_sql_only'](state['sql_query'])

        elif output_mode == "sql_with_results":
            state["formatted_response"] = format_funcs['format_sql_with_results'](
                state['sql_query'],
                str(state['sql_results'])
            )

        else:  # nlp_explanation
            llm = get_llm_func()
            formatter_prompt = get_prompt_func()

            chain = formatter_prompt | llm
            response = chain.invoke({
                "question": state["user_query"],
                "sql": state["sql_query"],
                "results": str(state["sql_results"]),
                "schema_context": str(state.get("schema_context", {}).get("evidence", []))
            })

            content = response.content if hasattr(response, 'content') else str(response)

            state["formatted_response"] = format_funcs['format_nlp_explanation'](
                state['user_query'],
                state['sql_query'],
                content,
                state['confidence_score']
            )

        return state

    return formatter_node


def create_error_handler_node(format_error_func):
    """Factory function to create error handler node"""

    def error_handler_node(state: SQLAgentState) -> SQLAgentState:
        """Handle errors and prepare error message"""
        state["formatted_response"] = format_error_func(
            state['user_query'],
            state.get('sql_query', 'Not generated'),
            state.get('error_count', 0)
        )
        return state

    return error_handler_node
