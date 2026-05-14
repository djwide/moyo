#!/usr/bin/env python3
"""
Example demonstrating Moyo's centralized configuration, structured logging, and Prometheus metrics.

This example shows:
1. Loading configuration from environment variables and YAML files
2. Using structured logging with context
3. Recording Prometheus metrics for pipeline operations
4. Starting and using the metrics server
"""

import time
import random
from pathlib import Path

from moyo.config.settings import get_settings, reload_settings
from moyo.logging import get_logger, setup_logging
from moyo.metrics import get_metrics_registry, pipeline_timer, document_processing_timer
from moyo.metrics_server import start_metrics_server, stop_metrics_server


def example_configuration():
    """Demonstrate configuration loading and usage."""
    print("🔧 Configuration Example")
    print("=" * 50)
    
    # Load default settings
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"Data directory: {settings.data_dir}")
    print(f"Embedding model: {settings.embedding.model_name}")
    print(f"FAISS index type: {settings.faiss.index_type}")
    print(f"Logging level: {settings.logging.level}")
    print(f"Prometheus enabled: {settings.prometheus.enabled}")
    
    # Show custom configuration
    if settings.custom_config:
        print(f"Custom config keys: {list(settings.custom_config.keys())}")
    
    print()


def example_structured_logging():
    """Demonstrate structured logging with context."""
    print("📝 Structured Logging Example")
    print("=" * 50)
    
    # Get a logger for this component
    logger = get_logger("example-component")
    
    # Basic logging
    logger.info("Starting example operation")
    
    # Logging with context
    logger.info("Processing document", 
               document_id="doc123", 
               document_type="pdf",
               file_size=1024000)
    
    # Using operation context for timing
    with logger.operation_context("data_processing", batch_size=1000):
        # Simulate some work
        time.sleep(0.1)
        logger.info("Processing batch", processed_items=500)
        
        # Simulate an error
        try:
            raise ValueError("Simulated processing error")
        except Exception as e:
            logger.exception("Error during processing", 
                           error_type=type(e).__name__,
                           retry_count=3)
    
    print("Check the logs for structured output!")
    print()


def example_prometheus_metrics():
    """Demonstrate Prometheus metrics recording."""
    print("📊 Prometheus Metrics Example")
    print("=" * 50)
    
    metrics_registry = get_metrics_registry()
    
    # Record pipeline operations
    with metrics_registry.pipeline_timer("example_pipeline", "demo"):
        time.sleep(0.2)
        print("Pipeline operation completed")
    
    # Record document processing
    metrics_registry.record_document_processed("demo", "pdf", "success")
    metrics_registry.record_document_processed("demo", "txt", "success")
    metrics_registry.record_document_processed("demo", "docx", "error")
    
    # Record FAISS operations
    metrics_registry.record_faiss_query("FlatL2", "knn", 0.05)
    metrics_registry.set_faiss_index_size("FlatL2", 10000)
    
    # Record embedding generation
    metrics_registry.record_embedding_generated("all-MiniLM-L6-v2", "success", 0.1)
    
    # Record LLM requests
    metrics_registry.record_llm_request("openai", "gpt-3.5-turbo", "success", 2.5)
    
    # Record fuzzing activities
    metrics_registry.record_fuzzing_campaign("basic", "success")
    metrics_registry.record_hypothesis_generated("llm", "success")
    metrics_registry.record_public_document_discovered("web", "crawling")
    
    print("Metrics recorded! Check the metrics endpoint to see them.")
    print()


@document_processing_timer("example", "mixed")
def example_document_processing():
    """Example document processing with metrics decorator."""
    print("📄 Document Processing Example")
    print("=" * 50)
    
    documents = [
        {"type": "pdf", "size": 1024000, "status": "success"},
        {"type": "txt", "size": 50000, "status": "success"},
        {"type": "docx", "size": 2048000, "status": "error"},
    ]
    
    metrics_registry = get_metrics_registry()
    
    for doc in documents:
        # Simulate processing
        time.sleep(0.1)
        
        # Record metrics
        metrics_registry.record_document_processed(
            "example", 
            doc["type"], 
            doc["status"]
        )
        
        print(f"Processed {doc['type']} document ({doc['size']} bytes) - {doc['status']}")
    
    print()


@pipeline_timer("example_campaign", "fuzzing")
def example_fuzzing_campaign():
    """Example fuzzing campaign with metrics."""
    print("🎯 Fuzzing Campaign Example")
    print("=" * 50)
    
    metrics_registry = get_metrics_registry()
    
    # Simulate campaign phases
    phases = ["initialization", "hypothesis_generation", "document_discovery", "analysis"]
    
    for phase in phases:
        with metrics_registry.pipeline_timer(f"campaign_{phase}", "fuzzing"):
            time.sleep(0.1)
            print(f"Completed phase: {phase}")
            
            # Record phase-specific metrics
            if phase == "hypothesis_generation":
                for i in range(5):
                    metrics_registry.record_hypothesis_generated("llm", "success")
            
            elif phase == "document_discovery":
                for i in range(3):
                    metrics_registry.record_public_document_discovered("web", "search")
    
    # Record overall campaign
    metrics_registry.record_fuzzing_campaign("comprehensive", "success")
    
    print("Fuzzing campaign completed!")
    print()


def example_metrics_server():
    """Demonstrate the metrics server."""
    print("🌐 Metrics Server Example")
    print("=" * 50)
    
    settings = get_settings()
    
    print(f"Starting metrics server on port {settings.prometheus.port}")
    print(f"Metrics endpoint: http://localhost:{settings.prometheus.port}{settings.prometheus.path}")
    print(f"Health endpoint: http://localhost:{settings.prometheus.port}/health")
    
    # Start the metrics server
    server = start_metrics_server(daemon=True)
    
    # Give it a moment to start
    time.sleep(1)
    
    if server.is_running():
        print("✅ Metrics server is running")
        
        # Generate some metrics
        example_prometheus_metrics()
        example_document_processing()
        example_fuzzing_campaign()
        
        print("\n📊 Current metrics:")
        metrics_data = get_metrics_registry().get_metrics()
        print(metrics_data[:500] + "..." if len(metrics_data) > 500 else metrics_data)
        
        # Stop the server
        print("\nStopping metrics server...")
        stop_metrics_server()
        print("✅ Metrics server stopped")
    else:
        print("❌ Failed to start metrics server")
    
    print()


def example_environment_configuration():
    """Demonstrate environment variable configuration."""
    print("🔧 Environment Configuration Example")
    print("=" * 50)
    
    print("You can override configuration using environment variables:")
    print("  MOYO_ENVIRONMENT=production")
    print("  MOYO_LOG_LEVEL=DEBUG")
    print("  MOYO_PROMETHEUS_PORT=9090")
    print("  MOYO_EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2")
    print("  MOYO_FAISS_INDEX_TYPE=IVF100")
    print()
    
    print("Or use a custom configuration file:")
    print("  moyo --config custom_config.yaml")
    print()


def main():
    """Run all examples."""
    print("🚀 Moyo Configuration and Monitoring Examples")
    print("=" * 60)
    print()
    
    # Setup logging
    setup_logging()
    
    # Run examples
    example_configuration()
    example_structured_logging()
    example_prometheus_metrics()
    example_document_processing()
    example_fuzzing_campaign()
    example_metrics_server()
    example_environment_configuration()
    
    print("✅ All examples completed!")
    print()
    print("Next steps:")
    print("1. Check the logs directory for structured log files")
    print("2. Use 'moyo metrics show' to view current metrics")
    print("3. Use 'moyo metrics start' to run the metrics server")
    print("4. Configure your own settings in config.yaml")


if __name__ == "__main__":
    main()
