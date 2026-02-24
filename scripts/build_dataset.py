#!/usr/bin/env python3
"""
Build instruction dataset from FastAPI markdown documentation.

This script:
1. Parses markdown files from data/raw/docs/
2. Extracts headings, code blocks, and content
3. Generates instruction-output pairs (1500-3000 examples)
4. Saves to data/dataset.jsonl

Dataset format:
    {"instruction": "<question>", "input": "", "output": "<structured answer>"}

Strategy:
- H1/H2 headings → "How do I...?" and "What is...?" questions
- Code blocks → "Quick example" sections
- Answers: one-line summary, short example, 2-4 bullets, 1 next-step bullet

Usage:
    python scripts/build_dataset.py
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_markdown_files(docs_dir: Path) -> List[Tuple[str, str]]:
    """Load all markdown files and return (filename, content) tuples."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documentation directory not found: {docs_dir}")
    
    md_files = list(docs_dir.rglob("*.md"))
    
    if not md_files:
        raise ValueError(f"No markdown files found in {docs_dir}")
    
    files_content = []
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            files_content.append((md_file.name, content))
        except Exception as e:
            print(f"⚠ Warning: Could not read {md_file}: {e}")
    
    print(f"✓ Loaded {len(files_content)} markdown files")
    return files_content


def extract_code_blocks(content: str) -> List[str]:
    """Extract code blocks from markdown content."""
    code_pattern = r'```(?:python|py|bash|shell|json|yaml|html)?\n(.*?)```'
    matches = re.findall(code_pattern, content, re.DOTALL)
    return [match.strip() for match in matches if match.strip()]


def extract_headings(content: str) -> List[Tuple[int, str]]:
    """Extract headings with their level (1-6) and text."""
    heading_pattern = r'^(#{1,6})\s+(.+)$'
    headings = []
    for line in content.split('\n'):
        match = re.match(heading_pattern, line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append((level, text))
    return headings


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove inline code markers
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def heading_to_question(heading: str, level: int) -> str:
    """Convert heading to instruction question."""
    heading_lower = heading.lower()
    
    # Skip certain headings that don't make good questions
    skip_patterns = ['table of contents', 'index', 'changelog', 'license']
    if any(pattern in heading_lower for pattern in skip_patterns):
        return None
    
    # Convert to question format
    if heading_lower.startswith(('how', 'what', 'why', 'when', 'where', 'which')):
        return heading
    elif '?' in heading:
        return heading
    
    # Add question prefix based on heading type
    if any(word in heading_lower for word in ['create', 'add', 'use', 'implement', 'build', 'make']):
        return f"How do I {heading.lower()}?"
    elif any(word in heading_lower for word in ['introduction', 'overview', 'about', 'concept']):
        return f"What is {heading.lower()}?"
    else:
        return f"What is {heading.lower()}?"


def build_answer_from_content(heading: str, content: str, code_blocks: List[str]) -> str:
    """Build structured answer from content and code blocks."""
    lines = []
    
    # One-line summary
    summary = clean_text(content[:200].split('\n')[0])
    if summary and len(summary) > 20:
        lines.append(summary)
    
    # Short example (if code blocks exist)
    if code_blocks:
        example_code = code_blocks[0]
        # Truncate long code blocks
        if len(example_code) > 300:
            example_code = example_code[:300] + "..."
        lines.append(f"\nQuick example:\n```python\n{example_code}\n```")
    
    # Extract bullet points from content
    bullet_pattern = r'^[-*]\s+(.+)$'
    bullets = []
    for line in content.split('\n'):
        match = re.match(bullet_pattern, line)
        if match:
            bullet_text = clean_text(match.group(1))
            if bullet_text and len(bullet_text) > 10:
                bullets.append(bullet_text)
                if len(bullets) >= 4:
                    break
    
    if bullets:
        lines.append("\nKey points:")
        for bullet in bullets[:4]:
            lines.append(f"- {bullet}")
    
    # Next step suggestion
    lines.append(f"\nNext step: Explore more about {heading.lower()} in the FastAPI documentation.")
    
    answer = "\n".join(lines)
    return answer.strip()


def process_markdown_files(files_content: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Process markdown files and generate instruction dataset."""
    dataset = []
    
    for filename, content in files_content:
        # Extract headings
        headings = extract_headings(content)
        
        # Extract code blocks
        code_blocks = extract_code_blocks(content)
        
        # Generate instruction-output pairs from headings
        for level, heading_text in headings:
            if level > 2:  # Focus on H1 and H2
                continue
            
            question = heading_to_question(heading_text, level)
            if not question:
                continue
            
            # Find content section for this heading
            # Simple approach: use content after heading
            heading_idx = content.find(f"#{'#' * (level-1)} {heading_text}")
            if heading_idx == -1:
                continue
            
            # Extract section content (next 1000 chars or until next heading)
            section_start = heading_idx + len(f"#{'#' * level} {heading_text}")
            section_end = section_start + 1000
            
            # Find next heading
            next_heading_pattern = r'^#{1,6}\s+'
            remaining = content[section_start:]
            next_match = re.search(next_heading_pattern, remaining, re.MULTILINE)
            if next_match:
                section_end = section_start + next_match.start()
            
            section_content = content[section_start:section_end]
            
            # Build answer
            answer = build_answer_from_content(heading_text, section_content, code_blocks)
            
            if len(answer) > 50:  # Ensure meaningful answers
                dataset.append({
                    "instruction": question,
                    "input": "",
                    "output": answer
                })
        
        # Also create examples from code blocks
        for i, code_block in enumerate(code_blocks[:3]):  # Limit per file
            if len(code_block) > 50:
                question = f"How do I use this FastAPI code example?"
                answer = f"Here's a code example:\n\n```python\n{code_block[:500]}\n```\n\nThis demonstrates FastAPI usage patterns."
                dataset.append({
                    "instruction": question,
                    "input": "",
                    "output": answer
                })
    
    return dataset


def filter_secrets(dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filter out potential secrets or sensitive information."""
    secret_patterns = [
        r'api[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}',
        r'password\s*[:=]\s*["\']?[^\s"\']{8,}',
        r'secret\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}',
        r'token\s*[:=]\s*["\']?[a-zA-Z0-9]{30,}',
    ]
    
    filtered = []
    for item in dataset:
        text = json.dumps(item).lower()
        if not any(re.search(pattern, text) for pattern in secret_patterns):
            filtered.append(item)
    
    removed = len(dataset) - len(filtered)
    if removed > 0:
        print(f"⚠ Filtered out {removed} examples with potential secrets")
    
    return filtered


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent
    docs_dir = base_dir / "data" / "raw" / "docs"
    output_file = base_dir / "data" / "dataset.jsonl"
    
    print("=" * 60)
    print("FastAPI Instruction Dataset Builder")
    print("=" * 60)
    print()
    
    # Ensure output directory exists
    ensure_dir(output_file.parent)
    
    # Load markdown files
    files_content = load_markdown_files(docs_dir)
    
    # Process files and build dataset
    print("📝 Processing markdown files and generating instructions...")
    dataset = process_markdown_files(files_content)
    
    # Filter secrets
    dataset = filter_secrets(dataset)
    
    # Ensure we have enough examples
    if len(dataset) < 1500:
        print(f"⚠ Warning: Only generated {len(dataset)} examples (target: 1500-3000)")
        print("  Consider checking documentation files or adjusting extraction logic.")
    elif len(dataset) > 3000:
        print(f"⚠ Warning: Generated {len(dataset)} examples (target: 1500-3000)")
        print("  Truncating to 3000 examples.")
        dataset = dataset[:3000]
    
    # Save dataset
    print(f"💾 Saving {len(dataset)} examples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 60)
    print(f"✓ Dataset build complete!")
    print(f"  Examples: {len(dataset)}")
    print(f"  Output: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
