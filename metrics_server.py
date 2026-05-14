"""Prometheus metrics server for Moyo project."""

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Optional

from .config.settings import get_settings
from .metrics import get_metrics_registry


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        settings = get_settings()
        metrics_registry = get_metrics_registry()
        
        if path == settings.prometheus.path:
            self._handle_metrics()
        elif path == "/health":
            self._handle_health()
        elif path == "/":
            self._handle_root()
        else:
            self._handle_not_found()
    
    def _handle_metrics(self):
        """Handle metrics endpoint."""
        try:
            metrics_data = get_metrics_registry().get_metrics()
            content_type = get_metrics_registry().get_metrics_content_type()
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(metrics_data)))
            self.end_headers()
            self.wfile.write(metrics_data.encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error generating metrics: {e}".encode('utf-8'))
    
    def _handle_health(self):
        """Handle health check endpoint."""
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "moyo-metrics"
        }
        
        import json
        response = json.dumps(health_data, indent=2)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def _handle_root(self):
        """Handle root endpoint."""
        settings = get_settings()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Moyo Metrics Server</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .endpoint {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                code {{ background: #e0e0e0; padding: 2px 4px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>Moyo Metrics Server</h1>
            <p>Available endpoints:</p>
            
            <div class="endpoint">
                <h3>Metrics</h3>
                <p><code>{settings.prometheus.path}</code> - Prometheus metrics</p>
            </div>
            
            <div class="endpoint">
                <h3>Health Check</h3>
                <p><code>/health</code> - Service health status</p>
            </div>
            
            <p><em>Server running on port {settings.prometheus.port}</em></p>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(html_content)))
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def _handle_not_found(self):
        """Handle 404 errors."""
        self.send_response(404)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Not Found")
    
    def log_message(self, format, *args):
        """Override to use structured logging."""
        # Use a simple log format to avoid cluttering metrics output
        pass


class MetricsServer:
    """Prometheus metrics server."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
    
    def start(self, daemon: bool = True):
        """Start the metrics server."""
        if self.running:
            return
        
        try:
            self.server = HTTPServer(
                ('localhost', self.settings.prometheus.port),
                MetricsHandler
            )
            
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=daemon
            )
            self.server_thread.start()
            
            self.running = True
            print(f"Metrics server started on http://localhost:{self.settings.prometheus.port}")
            print(f"Metrics available at http://localhost:{self.settings.prometheus.port}{self.settings.prometheus.path}")
            
        except Exception as e:
            print(f"Failed to start metrics server: {e}")
            raise
    
    def _run_server(self):
        """Run the HTTP server."""
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the metrics server."""
        if not self.running:
            return
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        self.running = False
        print("Metrics server stopped")
    
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.running


# Global metrics server instance
_metrics_server = None


def get_metrics_server() -> MetricsServer:
    """Get the global metrics server instance."""
    global _metrics_server
    if _metrics_server is None:
        _metrics_server = MetricsServer()
    return _metrics_server


def start_metrics_server(daemon: bool = True) -> MetricsServer:
    """Start the metrics server."""
    server = get_metrics_server()
    server.start(daemon=daemon)
    return server


def stop_metrics_server():
    """Stop the metrics server."""
    global _metrics_server
    if _metrics_server:
        _metrics_server.stop()
        _metrics_server = None


def run_metrics_server():
    """Run the metrics server in the foreground."""
    server = get_metrics_server()
    server.start(daemon=False)
    
    try:
        # Keep the main thread alive
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    run_metrics_server()
