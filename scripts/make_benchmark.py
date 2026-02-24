#!/usr/bin/env python3
"""
Create benchmark dataset with challenging questions for evaluation.

This script generates 100 challenging/edge-case questions about FastAPI
that are harder than typical instruction examples.

Format:
    {"id": "b001", "question": "...", "reference": ""}

Usage:
    python scripts/make_benchmark.py
"""

import json
from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def generate_benchmark_questions() -> list:
    """Generate challenging benchmark questions."""
    questions = [
        # Advanced routing and dependencies
        {"id": "b001", "question": "How do I create a dependency that depends on another dependency in FastAPI?", "reference": ""},
        {"id": "b002", "question": "What happens if a dependency raises an exception? How do I handle it?", "reference": ""},
        {"id": "b003", "question": "How do I override a dependency in FastAPI for testing?", "reference": ""},
        {"id": "b004", "question": "Can I use the same dependency multiple times in a single route? What are the implications?", "reference": ""},
        {"id": "b005", "question": "How do I create a route that accepts both query parameters and a request body?", "reference": ""},
        
        # Advanced request/response handling
        {"id": "b006", "question": "How do I handle file uploads with multiple files and metadata in FastAPI?", "reference": ""},
        {"id": "b007", "question": "What's the difference between Form() and Body() in FastAPI? When should I use each?", "reference": ""},
        {"id": "b008", "question": "How do I stream large responses without loading everything into memory?", "reference": ""},
        {"id": "b009", "question": "How do I handle custom HTTP status codes and response headers?", "reference": ""},
        {"id": "b010", "question": "Can I return different response models based on conditions in FastAPI?", "reference": ""},
        
        # Security and authentication
        {"id": "b011", "question": "How do I implement OAuth2 with password flow and refresh tokens in FastAPI?", "reference": ""},
        {"id": "b012", "question": "What's the difference between HTTPBearer and HTTPBasic in FastAPI security?", "reference": ""},
        {"id": "b013", "question": "How do I validate JWT tokens and extract user information in FastAPI?", "reference": ""},
        {"id": "b014", "question": "How do I implement rate limiting per user or IP address?", "reference": ""},
        {"id": "b015", "question": "How do I handle CORS for specific origins and methods in FastAPI?", "reference": ""},
        
        # Database and async
        {"id": "b016", "question": "How do I use async database sessions with SQLAlchemy in FastAPI?", "reference": ""},
        {"id": "b017", "question": "What's the best way to handle database connection pooling in FastAPI?", "reference": ""},
        {"id": "b018", "question": "How do I implement database transactions that span multiple endpoints?", "reference": ""},
        {"id": "b019", "question": "How do I handle database migrations in a FastAPI application?", "reference": ""},
        {"id": "b020", "question": "What's the difference between async and sync database operations in FastAPI?", "reference": ""},
        
        # Advanced Pydantic
        {"id": "b021", "question": "How do I create a Pydantic model with conditional validation based on another field?", "reference": ""},
        {"id": "b022", "question": "How do I use Pydantic validators to transform data before validation?", "reference": ""},
        {"id": "b023", "question": "Can I use Pydantic models with Generic types in FastAPI?", "reference": ""},
        {"id": "b024", "question": "How do I create nested models with optional fields and default values?", "reference": ""},
        {"id": "b025", "question": "How do I validate that a field matches a regex pattern in Pydantic?", "reference": ""},
        
        # WebSockets and real-time
        {"id": "b026", "question": "How do I handle WebSocket connections with authentication in FastAPI?", "reference": ""},
        {"id": "b027", "question": "How do I broadcast messages to multiple WebSocket clients?", "reference": ""},
        {"id": "b028", "question": "How do I handle WebSocket disconnections and cleanup?", "reference": ""},
        {"id": "b029", "question": "Can I use dependencies in WebSocket endpoints?", "reference": ""},
        {"id": "b030", "question": "How do I implement a WebSocket ping/pong mechanism?", "reference": ""},
        
        # Testing and deployment
        {"id": "b031", "question": "How do I test FastAPI endpoints that require authentication?", "reference": ""},
        {"id": "b032", "question": "How do I mock dependencies in FastAPI tests?", "reference": ""},
        {"id": "b033", "question": "How do I test WebSocket endpoints in FastAPI?", "reference": ""},
        {"id": "b034", "question": "What's the best way to structure a large FastAPI application with multiple routers?", "reference": ""},
        {"id": "b035", "question": "How do I handle environment-specific configuration in FastAPI?", "reference": ""},
        
        # Background tasks and async operations
        {"id": "b036", "question": "How do I run background tasks that outlive the request in FastAPI?", "reference": ""},
        {"id": "b037", "question": "What's the difference between BackgroundTasks and Celery in FastAPI?", "reference": ""},
        {"id": "b038", "question": "How do I handle long-running tasks with progress updates?", "reference": ""},
        {"id": "b039", "question": "How do I implement a task queue with FastAPI?", "reference": ""},
        {"id": "b040", "question": "How do I handle scheduled tasks in FastAPI?", "reference": ""},
        
        # Error handling and middleware
        {"id": "b041", "question": "How do I create custom exception handlers in FastAPI?", "reference": ""},
        {"id": "b042", "question": "How do I implement middleware that modifies request/response?", "reference": ""},
        {"id": "b043", "question": "How do I handle validation errors and return custom error messages?", "reference": ""},
        {"id": "b044", "question": "How do I log all requests and responses in FastAPI?", "reference": ""},
        {"id": "b045", "question": "How do I implement request timeout handling?", "reference": ""},
        
        # Advanced features
        {"id": "b046", "question": "How do I use FastAPI with GraphQL?", "reference": ""},
        {"id": "b047", "question": "How do I implement API versioning in FastAPI?", "reference": ""},
        {"id": "b048", "question": "How do I create reusable route handlers with shared logic?", "reference": ""},
        {"id": "b049", "question": "How do I implement request/response compression in FastAPI?", "reference": ""},
        {"id": "b050", "question": "How do I handle multipart form data with nested structures?", "reference": ""},
        
        # Edge cases and performance
        {"id": "b051", "question": "How do I handle very large JSON payloads in FastAPI?", "reference": ""},
        {"id": "b052", "question": "What's the maximum size for file uploads and how do I configure it?", "reference": ""},
        {"id": "b053", "question": "How do I implement request deduplication in FastAPI?", "reference": ""},
        {"id": "b054", "question": "How do I handle concurrent requests to the same resource safely?", "reference": ""},
        {"id": "b055", "question": "How do I optimize FastAPI for high-throughput scenarios?", "reference": ""},
        
        # Integration and external services
        {"id": "b056", "question": "How do I integrate FastAPI with Redis for caching?", "reference": ""},
        {"id": "b057", "question": "How do I use FastAPI with message queues like RabbitMQ?", "reference": ""},
        {"id": "b058", "question": "How do I implement health checks and readiness probes?", "reference": ""},
        {"id": "b059", "question": "How do I handle graceful shutdown in FastAPI?", "reference": ""},
        {"id": "b060", "question": "How do I implement request tracing and distributed tracing?", "reference": ""},
        
        # Advanced validation
        {"id": "b061", "question": "How do I validate that a date field is in the future?", "reference": ""},
        {"id": "b062", "question": "How do I create a custom validator that checks multiple fields together?", "reference": ""},
        {"id": "b063", "question": "How do I validate email addresses and phone numbers in Pydantic?", "reference": ""},
        {"id": "b064", "question": "How do I handle optional fields with complex validation rules?", "reference": ""},
        {"id": "b065", "question": "How do I validate JSON schemas dynamically at runtime?", "reference": ""},
        
        # Documentation and OpenAPI
        {"id": "b066", "question": "How do I customize the OpenAPI schema in FastAPI?", "reference": ""},
        {"id": "b067", "question": "How do I add examples to the OpenAPI documentation?", "reference": ""},
        {"id": "b068", "question": "How do I hide certain endpoints from the OpenAPI docs?", "reference": ""},
        {"id": "b069", "question": "How do I add custom tags and descriptions to endpoints?", "reference": ""},
        {"id": "b070", "question": "How do I generate OpenAPI schema for custom response types?", "reference": ""},
        
        # Advanced routing patterns
        {"id": "b071", "question": "How do I create routes with regex patterns in FastAPI?", "reference": ""},
        {"id": "b072", "question": "How do I handle route conflicts and precedence?", "reference": ""},
        {"id": "b073", "question": "How do I create a catch-all route that handles all unmatched paths?", "reference": ""},
        {"id": "b074", "question": "How do I implement route aliases and redirects?", "reference": ""},
        {"id": "b075", "question": "How do I create routes that match multiple HTTP methods?", "reference": ""},
        
        # State management
        {"id": "b076", "question": "How do I share state between requests in FastAPI?", "reference": ""},
        {"id": "b077", "question": "How do I implement request-scoped caching?", "reference": ""},
        {"id": "b078", "question": "How do I handle application-level configuration that changes at runtime?", "reference": ""},
        {"id": "b079", "question": "How do I implement feature flags in FastAPI?", "reference": ""},
        {"id": "b080", "question": "How do I manage application lifecycle events?", "reference": ""},
        
        # Advanced async patterns
        {"id": "b081", "question": "How do I handle async context managers in FastAPI dependencies?", "reference": ""},
        {"id": "b082", "question": "How do I implement async generators in FastAPI endpoints?", "reference": ""},
        {"id": "b083", "question": "How do I handle async cleanup and resource management?", "reference": ""},
        {"id": "b084", "question": "How do I implement async locks and semaphores in FastAPI?", "reference": ""},
        {"id": "b085", "question": "How do I handle async timeouts and cancellation?", "reference": ""},
        
        # Complex data structures
        {"id": "b086", "question": "How do I handle deeply nested JSON structures in FastAPI?", "reference": ""},
        {"id": "b087", "question": "How do I parse and validate XML in FastAPI?", "reference": ""},
        {"id": "b088", "question": "How do I handle binary data and base64 encoding?", "reference": ""},
        {"id": "b089", "question": "How do I work with geospatial data in FastAPI?", "reference": ""},
        {"id": "b090", "question": "How do I handle timezone-aware datetime objects?", "reference": ""},
        
        # Production concerns
        {"id": "b091", "question": "How do I implement request ID tracking across services?", "reference": ""},
        {"id": "b092", "question": "How do I handle database connection failures gracefully?", "reference": ""},
        {"id": "b093", "question": "How do I implement circuit breakers in FastAPI?", "reference": ""},
        {"id": "b094", "question": "How do I handle partial failures in batch operations?", "reference": ""},
        {"id": "b095", "question": "How do I implement idempotency keys for POST requests?", "reference": ""},
        
        # Advanced security
        {"id": "b096", "question": "How do I implement content security policy headers?", "reference": ""},
        {"id": "b097", "question": "How do I prevent SQL injection when using raw queries?", "reference": ""},
        {"id": "b098", "question": "How do I implement request signing and verification?", "reference": ""},
        {"id": "b099", "question": "How do I handle secrets management in FastAPI?", "reference": ""},
        {"id": "b100", "question": "How do I implement audit logging for sensitive operations?", "reference": ""},
    ]
    
    return questions


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / "data" / "benchmark.jsonl"
    
    print("=" * 60)
    print("FastAPI Benchmark Question Generator")
    print("=" * 60)
    print()
    
    # Ensure output directory exists
    ensure_dir(output_file.parent)
    
    # Generate questions
    print("📝 Generating benchmark questions...")
    questions = generate_benchmark_questions()
    
    # Save benchmark
    print(f"💾 Saving {len(questions)} questions to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in questions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 60)
    print(f"✓ Benchmark generation complete!")
    print(f"  Questions: {len(questions)}")
    print(f"  Output: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
