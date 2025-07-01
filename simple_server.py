#!/usr/bin/env python3
"""
Simple Web Server for Copilot Failure Analysis
Uses only built-in Python modules - no dependencies!
"""

import http.server
import socketserver
import json
import sqlite3
from urllib.parse import urlparse, parse_qs
import os
from datetime import datetime

PORT = 8080

def get_database_stats():
    """Get statistics from the SQLite database"""
    try:
        conn = sqlite3.connect('copilot_failures.db')
        cursor = conn.cursor()
        
        # Get basic counts
        cursor.execute("SELECT COUNT(*) FROM telemetry_events")
        total_events = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM failure_fingerprints")
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM clusters")
        total_clusters = cursor.fetchone()[0]
        
        # Get top failure patterns
        cursor.execute("""
            SELECT plugin_name, status_code, COUNT(*) as count
            FROM failure_fingerprints 
            WHERE plugin_name IS NOT NULL
            GROUP BY plugin_name, status_code 
            ORDER BY count DESC 
            LIMIT 10
        """)
        failure_patterns = cursor.fetchall()
        
        # Get recent events
        cursor.execute("""
            SELECT evaluation_id, customer_prompt, created_at
            FROM telemetry_events 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recent_events = cursor.fetchall()
        
        # Get cluster analysis results
        cursor.execute("""
            SELECT c.cluster_label, c.size, c.cluster_algorithm,
                   COUNT(ca.id) as assignment_count
            FROM clusters c
            LEFT JOIN cluster_assignments ca ON c.id = ca.cluster_id
            GROUP BY c.id, c.cluster_label, c.size, c.cluster_algorithm
            ORDER BY c.size DESC
        """)
        clusters = cursor.fetchall()
        
        # Get detailed cluster information with enhanced summaries
        cluster_details = []
        for cluster in clusters:
            cluster_label = cluster[0]
            cluster_size = cluster[1]
            cluster_algorithm = cluster[2]
            
            # Get cluster ID for this label
            cursor.execute("SELECT id FROM clusters WHERE cluster_label = ?", (cluster_label,))
            cluster_id_result = cursor.fetchone()
            if not cluster_id_result:
                continue
            cluster_id = cluster_id_result[0]
            
            # Get enhanced summary for this cluster
            cursor.execute("""
                SELECT summary_text, root_cause, common_plugins, common_endpoints
                FROM cluster_summaries 
                WHERE cluster_id = ?
            """, (cluster_id,))
            summary_result = cursor.fetchone()
            
            # Get sample events in this cluster
            cursor.execute("""
                SELECT te.evaluation_id, te.customer_prompt, 
                       fp.plugin_name, fp.status_code, fp.endpoint
                FROM cluster_assignments ca
                JOIN embeddings e ON ca.embedding_id = e.id
                JOIN telemetry_events te ON e.event_id = te.id
                LEFT JOIN failure_fingerprints fp ON te.id = fp.event_id
                WHERE ca.cluster_id = ?
                LIMIT 5
            """, (cluster_id,))
            sample_events = cursor.fetchall()
            
            # Get error patterns for this cluster
            cursor.execute("""
                SELECT fp.plugin_name, fp.status_code, COUNT(*) as count
                FROM cluster_assignments ca
                JOIN embeddings e ON ca.embedding_id = e.id
                JOIN telemetry_events te ON e.event_id = te.id
                JOIN failure_fingerprints fp ON te.id = fp.event_id
                WHERE ca.cluster_id = ?
                  AND fp.plugin_name IS NOT NULL
                GROUP BY fp.plugin_name, fp.status_code
                ORDER BY count DESC
            """, (cluster_id,))
            error_patterns = cursor.fetchall()
            
            cluster_details.append({
                'label': cluster_label,
                'size': cluster_size,
                'algorithm': cluster_algorithm,
                'sample_events': sample_events,
                'error_patterns': error_patterns,
                'summary': summary_result[0] if summary_result else None,
                'root_cause': summary_result[1] if summary_result else None,
                'common_plugins': summary_result[2] if summary_result else None,
                'common_endpoints': summary_result[3] if summary_result else None
            })
        
        conn.close()
        
        return {
            'total_events': total_events,
            'total_patterns': total_patterns,
            'total_embeddings': total_embeddings,
            'total_clusters': total_clusters,
            'failure_patterns': failure_patterns,
            'recent_events': recent_events,
            'clusters': clusters,
            'cluster_details': cluster_details
        }
        
    except Exception as e:
        print(f"Database error: {e}")
        return {
            'total_events': 0,
            'total_patterns': 0,
            'total_embeddings': 0,
            'total_clusters': 0,
            'failure_patterns': [],
            'recent_events': [],
            'clusters': [],
            'cluster_details': [],
            'error': str(e)
        }

class CopilotAnalysisHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_dashboard()
        elif parsed_path.path == '/api/data':
            self.serve_api_data()
        else:
            self.send_error(404, "File not found")
    
    def serve_dashboard(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Copilot Failure Analysis Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 4px solid #667eea; }
        .stat-number { font-size: 2.5em; font-weight: bold; color: #667eea; margin-bottom: 5px; }
        .stat-label { font-size: 1.1em; color: #666; text-transform: uppercase; letter-spacing: 1px; }
        .section { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .section h2 { color: #333; margin-bottom: 20px; font-size: 1.5em; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .cluster-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 15px; background: #fafafa; }
        .cluster-header { display: flex; justify-content: between; align-items: center; margin-bottom: 15px; }
        .cluster-title { font-size: 1.3em; font-weight: bold; color: #667eea; }
        .cluster-size { background: #667eea; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.9em; }
        .event-list { margin: 10px 0; }
        .event-item { padding: 8px 0; border-bottom: 1px solid #eee; font-size: 0.95em; }
        .error-patterns { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
        .error-tag { background: #ff6b6b; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; }
        .patterns-list { list-style: none; }
        .patterns-list li { padding: 10px 0; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
        .pattern-name { font-weight: 500; }
        .pattern-count { background: #667eea; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }
        .auto-refresh { text-align: center; margin-top: 20px; color: #666; font-style: italic; }
        .loading { text-align: center; padding: 50px; color: #666; }
        .error-message { background: #ffe6e6; border: 1px solid #ffcccc; color: #cc0000; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Copilot Failure Analysis</h1>
            <p>AI-Powered Clustering & Pattern Detection Dashboard</p>
        </div>
        
        <div id="content">
            <div class="loading">📊 Loading analysis results...</div>
        </div>
    </div>

    <script>
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                renderDashboard(data);
            } catch (error) {
                document.getElementById('content').innerHTML = 
                    '<div class="error-message">❌ Error loading data: ' + error.message + '</div>';
            }
        }

        function renderDashboard(data) {
            const content = document.getElementById('content');
            
            let html = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">${data.total_events}</div>
                        <div class="stat-label">📝 Total Events</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${data.total_patterns}</div>
                        <div class="stat-label">🔍 Failure Patterns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${data.total_embeddings}</div>
                        <div class="stat-label">🧠 Embeddings Generated</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${data.total_clusters}</div>
                        <div class="stat-label">🎯 Clusters Found</div>
                    </div>
                </div>
            `;

            // AI Cluster Analysis Section
            if (data.cluster_details && data.cluster_details.length > 0) {
                html += `
                    <div class="section">
                        <h2>🎯 AI Cluster Analysis Results</h2>
                        <p style="margin-bottom: 20px; color: #666;">Machine learning has identified ${data.total_clusters} distinct failure patterns:</p>
                `;
                
                data.cluster_details.forEach(cluster => {
                    html += `
                        <div class="cluster-card">
                            <div class="cluster-header">
                                <div class="cluster-title">Cluster ${cluster.label}</div>
                                <div class="cluster-size">${cluster.size} failures</div>
                            </div>
                    `;
                    
                    // Algorithm type
                    if (cluster.algorithm) {
                        html += `<p style="margin-bottom: 10px; color: #888; font-size: 0.9em;">📊 Algorithm: ${cluster.algorithm}</p>`;
                    }
                    
                    // Root Cause Analysis
                    if (cluster.root_cause) {
                        html += `
                            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #667eea;">
                                <h4 style="color: #667eea; margin-bottom: 8px;">🔍 Root Cause Analysis</h4>
                                <p style="color: #444; line-height: 1.4;">${cluster.root_cause}</p>
                            </div>
                        `;
                    }
                    
                    // AI Summary
                    if (cluster.summary) {
                        html += `
                            <div style="background: #f8fff0; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4CAF50;">
                                <h4 style="color: #4CAF50; margin-bottom: 8px;">🤖 AI Summary</h4>
                                <p style="color: #444; line-height: 1.4;">${cluster.summary}</p>
                            </div>
                        `;
                    }
                    
                    // Error patterns for this cluster
                    if (cluster.error_patterns && cluster.error_patterns.length > 0) {
                        html += '<div style="margin: 15px 0;"><strong>🏷️ Error Patterns:</strong><div class="error-patterns" style="margin-top: 8px;">';
                        cluster.error_patterns.forEach(pattern => {
                            html += `<span class="error-tag">${pattern[0]} ${pattern[1]} (${pattern[2]}×)</span>`;
                        });
                        html += '</div></div>';
                    }
                    
                    // Sample events
                    if (cluster.sample_events && cluster.sample_events.length > 0) {
                        html += '<div class="event-list" style="margin: 15px 0;"><strong>💬 Sample User Requests:</strong>';
                        cluster.sample_events.slice(0, 3).forEach(event => {
                            const promptShort = event[1].length > 80 ? event[1].substring(0, 80) + '...' : event[1];
                            html += `<div class="event-item" style="padding: 8px 0; border-bottom: 1px solid #eee;">"${promptShort}"</div>`;
                        });
                        html += '</div>';
                    }
                    
                    html += '</div>';
                });
                
                html += '</div>';
            }

            // Top Failure Patterns Section
            if (data.failure_patterns && data.failure_patterns.length > 0) {
                html += `
                    <div class="section">
                        <h2>🔥 Top Failure Patterns</h2>
                        <ul class="patterns-list">
                `;
                
                data.failure_patterns.forEach(pattern => {
                    html += `
                        <li>
                            <span class="pattern-name">🔌 ${pattern[0]} | ❌ ${pattern[1]}</span>
                            <span class="pattern-count">${pattern[2]}×</span>
                        </li>
                    `;
                });
                
                html += '</ul></div>';
            }

            // Recent Events Section
            if (data.recent_events && data.recent_events.length > 0) {
                html += `
                    <div class="section">
                        <h2>📋 Recent Failure Events</h2>
                        <div class="event-list">
                `;
                
                data.recent_events.slice(0, 5).forEach(event => {
                    const promptShort = event[1].length > 80 ? event[1].substring(0, 80) + '...' : event[1];
                    html += `
                        <div class="event-item">
                            <strong>[${event[0]}]</strong> ${promptShort}
                            <br><small style="color: #888;">⏰ ${event[2]}</small>
                        </div>
                    `;
                });
                
                html += '</div></div>';
            }

            html += `
                <div class="auto-refresh">
                    🔄 Auto-refreshes every 15 seconds | 
                    🌐 Dashboard: <strong>http://localhost:8080</strong> | 
                    📊 Run analysis: <strong>python run_ai_analysis.py</strong>
                </div>
            `;

            content.innerHTML = html;
        }

        // Load data on page load
        loadData();
        
        // Auto-refresh every 15 seconds
        setInterval(loadData, 15000);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_api_data(self):
        data = get_database_stats()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def main():
    # Check if database exists
    if not os.path.exists('copilot_failures.db'):
        print("⚠️  Database not found. Run 'python demo.py' first to create sample data.")
    
    print("🚀 Copilot Failure Analysis Dashboard")
    print("📊 Server running at: http://localhost:8080")
    print("🌐 Open your browser and visit the URL above!")
    print("🔄 Auto-refreshes every 15 seconds")
    print("⚡ Using built-in Python modules only (no dependencies!)")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", PORT), CopilotAnalysisHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped. Thanks for using Copilot Analysis!")

if __name__ == "__main__":
    main() 