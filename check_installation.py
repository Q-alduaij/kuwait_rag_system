#!/usr/bin/env python3

import importlib
import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported"""
    try:
        if import_name is None:
            import_name = package_name
        importlib.import_module(import_name)
        return True, f"✅ {package_name}"
    except ImportError as e:
        return False, f"❌ {package_name}: {e}"

def main():
    print("🔍 Checking Kuwait RAG System Dependencies...")
    print("=" * 50)
    
    # Critical packages
    critical_packages = [
        ("langchain", "langchain"),
        ("langchain-community", "langchain_community"),
        ("chromadb", "chromadb"),
        ("sentence-transformers", "sentence_transformers"),
        ("fastapi", "fastapi"),
        ("pydantic", "pydantic"),
        ("arabic-reshaper", "arabic_reshaper"),
        ("python-bidi", "bidi"),
    ]
    
    # Important packages
    important_packages = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("pypdf2", "PyPDF2"),
        ("python-docx", "docx"),
        ("beautifulsoup4", "bs4"),
        ("unstructured", "unstructured"),
        ("ragas", "ragas"),
        ("rank-bm25", "rank_bm25"),
    ]
    
    # Optional packages
    optional_packages = [
        ("camel-tools", "camel_tools"),
        ("pytesseract", "pytesseract"),
        ("pillow", "PIL"),
        ("prayer-times", "prayer_times"),
    ]
    
    print("\n📋 CRITICAL PACKAGES:")
    critical_ok = True
    for pkg_name, import_name in critical_packages:
        success, message = check_package(pkg_name, import_name)
        print(message)
        if not success:
            critical_ok = False
    
    print("\n📊 IMPORTANT PACKAGES:")
    for pkg_name, import_name in important_packages:
        success, message = check_package(pkg_name, import_name)
        print(message)
    
    print("\n🎯 OPTIONAL PACKAGES:")
    for pkg_name, import_name in optional_packages:
        success, message = check_package(pkg_name, import_name)
        print(message)
    
    print("\n" + "=" * 50)
    if critical_ok:
        print("🎉 All critical dependencies are installed!")
        print("🚀 Your Kuwait RAG system is ready to use!")
    else:
        print("❌ Some critical dependencies are missing.")
        print("💡 Run: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
