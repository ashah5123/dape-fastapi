#!/usr/bin/env python3
"""
Fetch FastAPI documentation from GitHub repository.

This script:
1. Clones the FastAPI repository (shallow clone) to data/raw/fastapi-repo
2. Copies markdown documentation files from docs/ to data/raw/docs/
3. Is idempotent (safe to run multiple times)

Usage:
    python scripts/fetch_docs.py
"""

import os
import shutil
import subprocess
from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory ensured: {path}")


def clone_fastapi_repo(repo_path: Path) -> None:
    """Clone FastAPI repository (shallow clone) if not already present."""
    if (repo_path / ".git").exists():
        print(f"✓ FastAPI repo already exists at {repo_path}")
        print("  To re-fetch, delete the directory first.")
        return
    
    repo_url = "https://github.com/tiangolo/fastapi.git"
    print(f"📥 Cloning FastAPI repository (shallow clone)...")
    print(f"   URL: {repo_url}")
    print(f"   Destination: {repo_path}")
    
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Successfully cloned FastAPI repository")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error cloning repository: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        raise


def copy_docs(source_docs: Path, dest_docs: Path) -> None:
    """Copy markdown documentation files from source to destination."""
    if not source_docs.exists():
        raise FileNotFoundError(f"Source docs directory not found: {source_docs}")
    
    ensure_dir(dest_docs)
    
    # Find all markdown files
    md_files = list(source_docs.rglob("*.md"))
    
    if not md_files:
        print(f"⚠ No markdown files found in {source_docs}")
        return
    
    copied_count = 0
    for md_file in md_files:
        # Preserve relative path structure
        relative_path = md_file.relative_to(source_docs)
        dest_file = dest_docs / relative_path
        
        # Create parent directories
        ensure_dir(dest_file.parent)
        
        # Copy file
        shutil.copy2(md_file, dest_file)
        copied_count += 1
    
    print(f"✓ Copied {copied_count} markdown files to {dest_docs}")


def main():
    """Main execution function."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_raw = base_dir / "data" / "raw"
    repo_path = data_raw / "fastapi-repo"
    source_docs = repo_path / "docs"
    dest_docs = data_raw / "docs"
    
    print("=" * 60)
    print("FastAPI Documentation Fetcher")
    print("=" * 60)
    print()
    
    # Ensure base directories exist
    ensure_dir(data_raw)
    
    # Clone repository
    clone_fastapi_repo(repo_path)
    
    # Copy documentation files
    if source_docs.exists():
        copy_docs(source_docs, dest_docs)
    else:
        print(f"⚠ Documentation directory not found: {source_docs}")
        print("  Repository may not have been cloned successfully.")
        return
    
    print()
    print("=" * 60)
    print("✓ Documentation fetch complete!")
    print(f"  Documentation available at: {dest_docs}")
    print("=" * 60)


if __name__ == "__main__":
    main()
