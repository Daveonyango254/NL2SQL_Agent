"""
One-time script to generate embeddings for all BIRD databases
Creates persistent Chroma vector stores to avoid regenerating embeddings on every query
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration - Use relative paths from script location
BASE_DIR = Path(__file__).parent.resolve()
BIRD_DB_PATH = BASE_DIR / "data" / "bird" / "dev_databases"
BIRD_DEV_JSON = BASE_DIR / "data" / "bird" / "dev_2_examples.json"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Ensure embeddings directory exists
EMBEDDINGS_DIR.mkdir(exist_ok=True)


def get_embeddings_model(use_local: bool = True):
    """Get embeddings model - local or OpenAI"""
    if use_local:
        try:
            print("Loading local embeddings model (all-MiniLM-L6-v2)...")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"Local embeddings failed: {e}. Using OpenAI embeddings.")
            return OpenAIEmbeddings()
    else:
        return OpenAIEmbeddings()


def load_csv_documents(db_id: str) -> List[Document]:
    """Load all CSV files for a database"""
    csv_dir = BIRD_DB_PATH / db_id / "database_description"
    documents = []

    if not csv_dir.exists():
        print(f"  [WARN] No CSV directory found for {db_id}")
        return documents

    for csv_path in csv_dir.glob("*.csv"):
        try:
            loader = CSVLoader(str(csv_path))
            docs = loader.load()

            # Add metadata
            for doc in docs:
                doc.metadata['source_file'] = csv_path.name
                doc.metadata['db_id'] = db_id
                doc.metadata['source_type'] = 'csv'

            documents.extend(docs)
            print(f"  [OK] Loaded {len(docs)} documents from {csv_path.name}")
        except Exception as e:
            print(f"  [FAIL] Failed to load {csv_path.name}: {e}")

    return documents


def load_dev_json_documents(db_id: str) -> List[Document]:
    """Load relevant examples from dev_2_examples.json for a specific database"""
    if not BIRD_DEV_JSON.exists():
        print(f"  [WARN] dev_2_examples.json not found at {BIRD_DEV_JSON}")
        return []

    try:
        with open(BIRD_DEV_JSON, 'r', encoding='utf-8') as f:
            all_examples = json.load(f)

        # Filter examples for this database
        db_examples = [ex for ex in all_examples if ex.get('db_id') == db_id]

        documents = []
        for example in db_examples:
            # Create a rich text representation of the example
            content = f"""
Question: {example.get('question', '')}
SQL Query: {example.get('SQL', '')}
Evidence: {example.get('evidence', '')}
Difficulty: {example.get('difficulty', 'unknown')}
Database: {db_id}
"""
            doc = Document(
                page_content=content.strip(),
                metadata={
                    'db_id': db_id,
                    'source_type': 'dev_json',
                    'question_id': example.get('question_id', ''),
                    'difficulty': example.get('difficulty', 'unknown')
                }
            )
            documents.append(doc)

        print(f"  [OK] Loaded {len(documents)} examples from dev_2_examples.json")
        return documents

    except Exception as e:
        print(f"  [FAIL] Failed to load dev_2_examples.json: {e}")
        return []


def create_embeddings_for_database(db_id: str, use_local_embeddings: bool = True):
    """Create and persist embeddings for a single database"""
    print(f"\n{'='*60}")
    print(f"Processing database: {db_id}")
    print(f"{'='*60}")

    # Load documents from all sources
    all_documents = []

    # Load CSV files
    csv_docs = load_csv_documents(db_id)
    all_documents.extend(csv_docs)

    # Load dev_2_examples.json examples
    json_docs = load_dev_json_documents(db_id)
    all_documents.extend(json_docs)

    if not all_documents:
        print(f"  [WARN] No documents found for {db_id}, skipping...")
        return False

    print(f"\n  Total documents: {len(all_documents)}")
    print(f"  - CSV documents: {len(csv_docs)}")
    print(f"  - dev_2_examples.json examples: {len(json_docs)}")

    # Split documents
    print("\n  Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_documents)
    print(f"  [OK] Created {len(splits)} chunks")

    # Create embeddings
    print(f"\n  Generating embeddings...")
    embeddings = get_embeddings_model(use_local=use_local_embeddings)

    # Create persistent vector store
    persist_directory = str(EMBEDDINGS_DIR / db_id)
    print(f"  Saving to: {persist_directory}")

    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=f"{db_id}_collection"
        )

        print(f"  [OK] Embeddings created and persisted successfully!")
        print(f"  Collection: {db_id}_collection")
        print(f"  Vector count: {len(splits)}")

        return True

    except Exception as e:
        print(f"  [FAIL] Failed to create embeddings: {e}")
        return False


def discover_databases() -> List[str]:
    """Discover all available database IDs"""
    databases = []
    if BIRD_DB_PATH.exists():
        for db_dir in BIRD_DB_PATH.iterdir():
            if db_dir.is_dir():
                sqlite_file = db_dir / f"{db_dir.name}.sqlite"
                if sqlite_file.exists():
                    databases.append(db_dir.name)
    return sorted(databases)


def main(use_local_embeddings: bool = True, specific_db: str = None):
    """Main function to generate embeddings for all or specific databases"""
    print("\n" + "="*60)
    print("BIRD Database Embeddings Generator")
    print("="*60)
    print(f"Embeddings directory: {EMBEDDINGS_DIR}")
    print(
        f"Using: {'Local HuggingFace' if use_local_embeddings else 'OpenAI'} embeddings")
    print("="*60)

    # Discover databases
    if specific_db:
        databases = [specific_db]
        print(f"\nProcessing specific database: {specific_db}")
    else:
        databases = discover_databases()
        print(f"\nFound {len(databases)} databases to process:")
        for db in databases:
            print(f"  - {db}")

    if not databases:
        print("\n[WARN] No databases found!")
        return

    # Process each database
    success_count = 0
    fail_count = 0

    for i, db_id in enumerate(databases, 1):
        print(f"\n[{i}/{len(databases)}] Processing {db_id}...")
        success = create_embeddings_for_database(db_id, use_local_embeddings)

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total databases: {len(databases)}")
    print(f"[OK] Successful: {success_count}")
    print(f"[FAIL] Failed: {fail_count}")
    print(f"\nEmbeddings saved to: {EMBEDDINGS_DIR}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings for BIRD databases"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Specific database to process (default: all)"
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI embeddings instead of local"
    )

    args = parser.parse_args()

    main(
        use_local_embeddings=not args.use_openai,
        specific_db=args.db
    )
