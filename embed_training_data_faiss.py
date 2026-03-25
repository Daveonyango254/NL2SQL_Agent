"""
Generate embeddings using FAISS instead of Chroma (NumPy 2.0 compatible)
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import argparse

load_dotenv()

from api.src.utils.database import get_base_dir

BASE_DIR = get_base_dir()
BIRD_DB_PATH = BASE_DIR / "data" / "bird" / "dev_databases"
BIRD_DEV_JSON = BASE_DIR / "data" / "bird" / "dev_2_examples.json"
EMBEDDINGS_DIR = BASE_DIR / "embeddings_faiss"

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
    """Load relevant examples from dev_2_examples.json"""
    if not BIRD_DEV_JSON.exists():
        print(f"  [WARN] dev_2_examples.json not found")
        return []

    try:
        with open(BIRD_DEV_JSON, 'r', encoding='utf-8') as f:
            all_examples = json.load(f)

        db_examples = [ex for ex in all_examples if ex.get('db_id') == db_id]

        documents = []
        for example in db_examples:
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
                    'source_type': 'dev_example',
                    'db_id': db_id,
                    'difficulty': example.get('difficulty', 'unknown')
                }
            )
            documents.append(doc)

        print(f"  [OK] Loaded {len(documents)} examples from dev_2_examples.json")
        return documents

    except Exception as e:
        print(f"  [FAIL] Failed to load dev_2_examples.json: {e}")
        return []


def process_database(db_id: str, embeddings_model, use_local: bool = True):
    """Process a single database and create FAISS vectorstore"""
    print(f"\n{'='*60}")
    print(f"Processing database: {db_id}")
    print('='*60)

    # Load documents
    csv_docs = load_csv_documents(db_id)
    json_docs = load_dev_json_documents(db_id)

    all_documents = csv_docs + json_docs

    print(f"\n  Total documents: {len(all_documents)}")
    print(f"  - CSV documents: {len(csv_docs)}")
    print(f"  - dev_2_examples.json examples: {len(json_docs)}")

    if not all_documents:
        print(f"  [WARN] No documents found for {db_id}")
        return False

    # Split documents
    print(f"\n  Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"  [OK] Created {len(chunks)} chunks")

    # Create FAISS vectorstore
    print(f"\n  Generating embeddings with FAISS...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings_model)

        # Save vectorstore
        save_path = EMBEDDINGS_DIR / db_id
        save_path.mkdir(exist_ok=True)

        vectorstore.save_local(str(save_path))

        print(f"  [OK] Saved FAISS index to: {save_path}")
        return True

    except Exception as e:
        print(f"  [FAIL] Failed to create embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate FAISS embeddings for BIRD databases')
    parser.add_argument('--db', type=str, help='Specific database to process (default: all)')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI embeddings instead of local')
    args = parser.parse_args()

    print("="*60)
    print("BIRD Database Embeddings Generator (FAISS)")
    print("="*60)
    print(f"Embeddings directory: {EMBEDDINGS_DIR}")

    use_local = not args.use_openai
    print(f"Using: {'Local HuggingFace' if use_local else 'OpenAI'} embeddings")
    print("="*60)

    # Get embeddings model
    embeddings_model = get_embeddings_model(use_local)

    # Get list of databases
    if args.db:
        databases = [args.db]
        print(f"\nProcessing specific database: {args.db}")
    else:
        databases = sorted([d.name for d in BIRD_DB_PATH.iterdir() if d.is_dir()])
        print(f"\nFound {len(databases)} databases to process:")
        for db in databases:
            print(f"  - {db}")

    # Process each database
    successful = 0
    failed = 0

    for i, db_id in enumerate(databases, 1):
        print(f"\n[{i}/{len(databases)}] Processing {db_id}...")
        if process_database(db_id, embeddings_model, use_local):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total databases: {len(databases)}")
    print(f"[OK] Successful: {successful}")
    print(f"[FAIL] Failed: {failed}")
    print(f"\nEmbeddings saved to: {EMBEDDINGS_DIR}")
    print('='*60)


if __name__ == "__main__":
    main()
