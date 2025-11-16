#!/usr/bin/env python3
"""
RegNavigator System Diagnostic
Checks all components and reports status
"""

import os
import sys
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def check(condition, message):
    """Print check result"""
    if condition:
        print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")
        return True
    else:
        print(f"{Colors.RED}❌ {message}{Colors.RESET}")
        return False

def warn(message):
    """Print warning"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")

def info(message):
    """Print info"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")

def main():
    print(f"\n{Colors.BLUE}{'='*60}")
    print("RegNavigator System Diagnostic")
    print(f"{'='*60}{Colors.RESET}\n")
    
    issues = []
    
    # 1. Check .env file
    print(f"{Colors.BLUE}[1] Environment Configuration{Colors.RESET}")
    env_path = Path(".env")
    if check(env_path.exists(), f".env file exists at {env_path.absolute()}"):
        # Load and check keys
        from dotenv import load_dotenv
        load_dotenv(env_path)
        
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        provider = os.getenv("LLM_PROVIDER", "openai")
        
        info(f"LLM Provider: {provider}")
        
        if provider == "openai":
            if check(bool(openai_key) and not openai_key.startswith("sk-proj-xxx"), "OPENAI_API_KEY is set"):
                info(f"Key preview: {openai_key[:15]}...")
            else:
                issues.append("OPENAI_API_KEY not properly set in .env")
        
        elif provider == "anthropic":
            if check(bool(anthropic_key) and not anthropic_key.startswith("sk-ant-xxx"), "ANTHROPIC_API_KEY is set"):
                info(f"Key preview: {anthropic_key[:15]}...")
            else:
                issues.append("ANTHROPIC_API_KEY not properly set in .env")
    else:
        issues.append(".env file not found - copy .env.example to .env and add your API key")
    
    print()
    
    # 2. Check Python dependencies
    print(f"{Colors.BLUE}[2] Python Dependencies{Colors.RESET}")
    deps = [
        ("fastapi", "FastAPI"),
        ("chromadb", "ChromaDB"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("FlagEmbedding", "BGE Models"),
        ("openai", "OpenAI SDK"),
        ("anthropic", "Anthropic SDK"),
        ("dotenv", "python-dotenv")
    ]
    
    for module, name in deps:
        try:
            __import__(module)
            check(True, f"{name} installed")
        except ImportError:
            check(False, f"{name} NOT installed")
            issues.append(f"Install {name}: pip install {module}")
    
    print()
    
    # 3. Check data directory
    print(f"{Colors.BLUE}[3] Data Directory{Colors.RESET}")
    data_root = Path("data")
    if check(data_root.exists(), f"Data directory exists: {data_root.absolute()}"):
        
        # Find PDFs
        pdf_files = list(data_root.rglob("*.pdf"))
        if check(len(pdf_files) > 0, f"Found {len(pdf_files)} PDF files"):
            for pdf in pdf_files[:3]:
                info(f"  {pdf.relative_to(data_root)}")
            if len(pdf_files) > 3:
                info(f"  ... and {len(pdf_files) - 3} more")
        else:
            warn("No PDF files found in data/ directory")
            info("Expected structure: data/CA/pdfs/*.pdf")
    else:
        issues.append("data/ directory not found - create it and add PDFs")
    
    print()
    
    # 4. Check vector stores
    print(f"{Colors.BLUE}[4] Vector Stores{Colors.RESET}")
    chroma_dir = Path("chroma_store")
    bm25_dir = Path("bm25_store")
    
    check(chroma_dir.exists(), f"ChromaDB store: {chroma_dir.absolute()}")
    check(bm25_dir.exists(), f"BM25 store: {bm25_dir.absolute()}")
    
    # Check if populated
    if chroma_dir.exists():
        subdirs = list(chroma_dir.iterdir())
        if len(subdirs) > 0:
            info(f"ChromaDB appears populated ({len(subdirs)} items)")
        else:
            warn("ChromaDB is empty - run: python -m regnavigator.ingest")
    
    if bm25_dir.exists():
        pkl_files = list(bm25_dir.glob("*.pkl"))
        if len(pkl_files) > 0:
            info(f"BM25 has {len(pkl_files)} index(es)")
        else:
            warn("BM25 is empty - run: python -m regnavigator.ingest")
    
    print()
    
    # 5. Test imports
    print(f"{Colors.BLUE}[5] Module Imports{Colors.RESET}")
    try:
        from regnavigator.embeddings import EmbeddingModel
        check(True, "embeddings module")
    except Exception as e:
        check(False, f"embeddings module: {e}")
        issues.append("Fix embeddings module")
    
    try:
        from regnavigator.llm_providers import get_provider, validate_llm_config
        check(True, "llm_providers module")
        
        # Check LLM status
        llm_status = validate_llm_config()
        if llm_status["ready"]:
            check(True, "LLM configuration ready")
        else:
            check(False, "LLM configuration NOT ready")
            issues.append(f"LLM issue: {llm_status}")
    except Exception as e:
        check(False, f"llm_providers module: {e}")
        issues.append("Fix llm_providers module")
    
    try:
        from regnavigator.retriever import HybridRetriever
        check(True, "retriever module")
    except Exception as e:
        check(False, f"retriever module: {e}")
        issues.append("Fix retriever module")
    
    print()
    
    # 6. Summary
    print(f"{Colors.BLUE}{'='*60}")
    print("Summary")
    print(f"{'='*60}{Colors.RESET}\n")
    
    if issues:
        print(f"{Colors.RED}❌ Found {len(issues)} issue(s):{Colors.RESET}\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print(f"{Colors.YELLOW}Fix these issues and run diagnostic again.{Colors.RESET}")
        return 1
    else:
        print(f"{Colors.GREEN}✅ All checks passed! System is ready.{Colors.RESET}\n")
        print("Next steps:")
        print("  ➤ To (re)index regulations:")
        print("      python -m regnavigator.ingest")

        print("  ➤ To run the backend API:")
        print("      python main.py")

        print("  ➤ To use the interface:")
        print("      Open index.html in your browser")
        return 0

if __name__ == "__main__":
    sys.exit(main())
