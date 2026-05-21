"""Prometheus metrics for moyo project."""

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
from functools import wraps

from .config.settings import get_settings


# Global metrics registry
_metrics_registry = {}


class MetricsRegistry:
    """Registry for Prometheus metrics."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.metrics = {}
        self._initialized = False
    
    def _init_prometheus(self):
        """Initialize Prometheus client if enabled."""
        if self._initialized or not self.settings.prometheus.enabled:
            return
        
        try:
            from prometheus_client import (
                Counter, Histogram, Gauge, Summary, 
                generate_latest, CONTENT_TYPE_LATEST,
                CollectorRegistry, multiprocess
            )
            
            # Create registry
            if self.settings.environment == 'production':
                self.registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(self.registry)
            else:
                from prometheus_client import REGISTRY
                self.registry = REGISTRY
            
            # Pipeline metrics
            self.pipeline_duration = Histogram(
                'pipeline_duration_seconds',
                'Pipeline operation duration in seconds',
                ['operation', 'component', 'status'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.pipeline_operations_total = Counter(
                'pipeline_operations_total',
                'Total number of pipeline operations',
                ['operation', 'component', 'status'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            # Document processing metrics
            self.documents_processed_total = Counter(
                'documents_processed_total',
                'Total number of documents processed',
                ['component', 'document_type', 'status'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.document_processing_duration = Histogram(
                'document_processing_duration_seconds',
                'Document processing duration in seconds',
                ['component', 'document_type'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            # FAISS index metrics
            self.faiss_queries_total = Counter(
                'faiss_queries_total',
                'Total number of FAISS queries',
                ['index_type', 'query_type'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.faiss_query_duration = Histogram(
                'faiss_query_duration_seconds',
                'FAISS query duration in seconds',
                ['index_type', 'query_type'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.faiss_index_size = Gauge(
                'faiss_index_size',
                'Number of vectors in FAISS index',
                ['index_type'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            # Embedding metrics
            self.embeddings_generated_total = Counter(
                'embeddings_generated_total',
                'Total number of embeddings generated',
                ['model', 'status'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.embedding_generation_duration = Histogram(
                'embedding_generation_duration_seconds',
                'Embedding generation duration in seconds',
                ['model'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            # LLM metrics
            self.llm_requests_total = Counter(
                'llm_requests_total',
                'Total number of LLM requests',
                ['provider', 'model', 'status'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.llm_request_duration = Histogram(
                'llm_request_duration_seconds',
                'LLM request duration in seconds',
                ['provider', 'model'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            # Fuzzing metrics
            self.fuzzing_campaigns_total = Counter(
                'fuzzing_campaigns_total',
                'Total number of fuzzing campaigns',
                ['campaign_type', 'status'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.hypotheses_generated_total = Counter(
                'hypotheses_generated_total',
                'Total number of hypotheses generated',
                ['generator_type', 'status'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.public_documents_discovered_total = Counter(
                'public_documents_discovered_total',
                'Total number of public documents discovered',
                ['source', 'discovery_method'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            # System metrics
            self.memory_usage_bytes = Gauge(
                'memory_usage_bytes',
                'Memory usage in bytes',
                ['component'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self.cpu_usage_percent = Gauge(
                'cpu_usage_percent',
                'CPU usage percentage',
                ['component'],
                namespace=self.settings.prometheus.namespace,
                subsystem=self.settings.prometheus.subsystem,
                registry=self.registry
            )
            
            self._initialized = True
            
        except ImportError:
            print("Warning: prometheus_client not installed; metrics will be disabled.")
            self._initialized = False
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        self._init_prometheus()
        
        if not self._initialized:
            return "# Metrics disabled - prometheus_client not available"
        
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            return f"# Error generating metrics: {e}"
    
    def get_metrics_content_type(self) -> str:
        """Get content type for metrics endpoint."""
        try:
            from prometheus_client import CONTENT_TYPE_LATEST
            return CONTENT_TYPE_LATEST
        except ImportError:
            return "text/plain"
    
    @contextmanager
    def pipeline_timer(self, operation: str, component: str = "unknown"):
        """Context manager for timing pipeline operations."""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self._record_pipeline_metrics(operation, component, status, duration)
    
    def _record_pipeline_metrics(self, operation: str, component: str, status: str, duration: float):
        """Record pipeline metrics."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.pipeline_duration.labels(
                operation=operation,
                component=component,
                status=status
            ).observe(duration)
            
            self.pipeline_operations_total.labels(
                operation=operation,
                component=component,
                status=status
            ).inc()
        except Exception as e:
            print(f"Warning: Failed to record pipeline metrics: {e}")
    
    def record_document_processed(self, component: str, document_type: str, status: str = "success"):
        """Record document processing."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.documents_processed_total.labels(
                component=component,
                document_type=document_type,
                status=status
            ).inc()
        except Exception as e:
            print(f"Warning: Failed to record document metrics: {e}")
    
    @contextmanager
    def document_processing_timer(self, component: str, document_type: str):
        """Context manager for timing document processing."""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._record_document_processing_duration(component, document_type, duration)
    
    def _record_document_processing_duration(self, component: str, document_type: str, duration: float):
        """Record document processing duration."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.document_processing_duration.labels(
                component=component,
                document_type=document_type
            ).observe(duration)
        except Exception as e:
            print(f"Warning: Failed to record document processing duration: {e}")
    
    def record_faiss_query(self, index_type: str, query_type: str, duration: float):
        """Record FAISS query metrics."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.faiss_queries_total.labels(
                index_type=index_type,
                query_type=query_type
            ).inc()
            
            self.faiss_query_duration.labels(
                index_type=index_type,
                query_type=query_type
            ).observe(duration)
        except Exception as e:
            print(f"Warning: Failed to record FAISS metrics: {e}")
    
    def set_faiss_index_size(self, index_type: str, size: int):
        """Set FAISS index size."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.faiss_index_size.labels(index_type=index_type).set(size)
        except Exception as e:
            print(f"Warning: Failed to set FAISS index size: {e}")
    
    def record_embedding_generated(self, model: str, status: str = "success", duration: float = None):
        """Record embedding generation metrics."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.embeddings_generated_total.labels(
                model=model,
                status=status
            ).inc()
            
            if duration is not None:
                self.embedding_generation_duration.labels(model=model).observe(duration)
        except Exception as e:
            print(f"Warning: Failed to record embedding metrics: {e}")
    
    def record_llm_request(self, provider: str, model: str, status: str, duration: float):
        """Record LLM request metrics."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.llm_requests_total.labels(
                provider=provider,
                model=model,
                status=status
            ).inc()
            
            self.llm_request_duration.labels(
                provider=provider,
                model=model
            ).observe(duration)
        except Exception as e:
            print(f"Warning: Failed to record LLM metrics: {e}")
    
    def record_fuzzing_campaign(self, campaign_type: str, status: str):
        """Record fuzzing campaign metrics."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.fuzzing_campaigns_total.labels(
                campaign_type=campaign_type,
                status=status
            ).inc()
        except Exception as e:
            print(f"Warning: Failed to record fuzzing campaign metrics: {e}")
    
    def record_hypothesis_generated(self, generator_type: str, status: str):
        """Record hypothesis generation metrics."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.hypotheses_generated_total.labels(
                generator_type=generator_type,
                status=status
            ).inc()
        except Exception as e:
            print(f"Warning: Failed to record hypothesis metrics: {e}")
    
    def record_public_document_discovered(self, source: str, discovery_method: str):
        """Record public document discovery metrics."""
        self._init_prometheus()
        
        if not self._initialized:
            return
        
        try:
            self.public_documents_discovered_total.labels(
                source=source,
                discovery_method=discovery_method
            ).inc()
        except Exception as e:
            print(f"Warning: Failed to record document discovery metrics: {e}")


# Global metrics registry instance
_metrics_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _metrics_registry


def pipeline_timer(operation: str, component: str = "unknown"):
    """Decorator for timing pipeline operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _metrics_registry.pipeline_timer(operation, component):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def document_processing_timer(component: str, document_type: str):
    """Decorator for timing document processing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _metrics_registry.document_processing_timer(component, document_type):
                return func(*args, **kwargs)
        return wrapper
    return decorator
