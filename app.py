import streamlit as st
import sys
from pathlib import Path
import json
import yaml
from SQL_Agent import run_sql_agent, discover_databases, get_database_path, CONFIG
# from SQL_Agent_SLM import run_sql_agent, discover_databases, get_database_path   # For SLM version

# Page configuration
st.set_page_config(
    page_title="Agentic NL2SQL Agent",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Agentic NL2SQL Agent")
st.markdown(
    "Convert natural language questions into SQL queries and get instant results")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Discover available databases
available_databases = discover_databases()
db_list = list(available_databases.keys())

if not db_list:
    st.error("No databases found in the data directory")
    st.stop()

# Database selection
selected_db = st.sidebar.selectbox(
    "Select Database",
    options=db_list,
    index=0,
    help="Choose which database to query"
)

# Output mode selection
output_mode = st.sidebar.radio(
    "Output Mode",
    options=["nlp_explanation", "sql_with_results", "sql_only"],
    index=0,
    help="Choose how you want the results displayed"
)

# Display database info
with st.sidebar.expander("📊 Database Info"):
    db_path = get_database_path(selected_db)
    if db_path and db_path.exists():
        st.success(f"✓ Database found")
        st.text(f"Path: {db_path.name}")
    else:
        st.warning("Database file not found")

# Display model configuration
with st.sidebar.expander("🤖 Model Configuration"):
    primary_type = CONFIG.get('primary_model_type', 'openai')

    if primary_type == 'ollama':
        st.info("🦙 Using Ollama (Local)")
        st.text(f"Model: {CONFIG['ollama'].get('sql_generator_model', 'N/A')}")
        st.text(f"Base URL: {CONFIG['ollama'].get('base_url', 'N/A')}")
    else:
        st.info("🌐 Using OpenAI")
        st.text(f"Model: {CONFIG['openai'].get('sql_generator_model', 'N/A')}")

    st.text(f"Fallback: {CONFIG['openai'].get('fallback_model', 'N/A')}")
    st.text(f"Max Retries: {CONFIG['retry'].get('max_retries', 3)}")
    st.caption("Configure in config.yaml")

# Main interface
st.markdown("---")

# Query input
user_query = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g., What are the top 5 schools with the highest enrollment?",
    help="Ask a question in natural language about the selected database"
)

# Example queries (optional)
with st.expander("💡 Example Questions"):
    st.markdown("""
    **California Schools:**
    - List all charter schools in the database
    - What is the average enrollment across all schools?
    - Which schools have the highest free meal rates?

    **Formula 1:**
    - Who won the most races in 2021?
    - List all drivers from Ferrari
    - What are the fastest lap times at Monaco?
    """)

# Run query button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    run_button = st.button("🚀 Run Query", type="primary",
                           use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Results section
if run_button:
    if not user_query.strip():
        st.warning("⚠️ Please enter a question")
    else:
        # Show loading spinner
        with st.spinner("🔄 Processing your query..."):
            try:
                # Run the SQL agent
                result = run_sql_agent(
                    query=user_query,
                    db_id=selected_db,
                    output_mode=output_mode
                )

                # Display results
                st.markdown("---")
                st.subheader("📋 Results")
                st.markdown(result)

                # Success message
                st.success("✅ Query completed successfully!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)

# Footer
st.markdown("---")

# Dynamic footer based on config
primary_type = CONFIG.get('primary_model_type', 'openai')
if primary_type == 'ollama':
    model_info = f"Ollama ({CONFIG['ollama'].get('sql_generator_model', 'local')})"
else:
    model_info = f"OpenAI ({CONFIG['openai'].get('sql_generator_model', 'gpt-4o')})"

st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
    Built with ❤️ using Streamlit | Powered by {model_info} + LangChain
    </div>
    """,
    unsafe_allow_html=True
)
