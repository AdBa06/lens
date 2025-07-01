#!/usr/bin/env python3
"""
Simple Web Dashboard for Copilot Failure Analysis System
Shows analysis results in a web browser
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
from typing import Dict, List, Any
from database import SessionLocal
from models import TelemetryEvent, FailureFingerprint, Embedding, Cluster
import uvicorn

app = FastAPI(title="Copilot Failure Analysis Dashboard")

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics"""
    db = SessionLocal()
    try:
        stats = {
            "total_events": db.query(TelemetryEvent).count(),
            "total_fingerprints": db.query(FailureFingerprint).count(),
            "total_embeddings": db.query(Embedding).count(),
            "total_clusters": db.query(Cluster).count()
        }
        return stats
    finally:
        db.close()

def get_failure_patterns() -> List[Dict[str, Any]]:
    """Get failure pattern statistics"""
    db = SessionLocal()
    try:
        patterns = {}
        fingerprints = db.query(FailureFingerprint).all()
        
        for fp in fingerprints:
            key = f"{fp.plugin_name}/{fp.error_type}"
            if key not in patterns:
                patterns[key] = {
                    "pattern": key,
                    "plugin": fp.plugin_name,
                    "error_type": fp.error_type,
                    "count": 0,
                    "endpoints": set()
                }
            patterns[key]["count"] += 1
            if fp.endpoint:
                patterns[key]["endpoints"].add(fp.endpoint)
        
        # Convert sets to lists for JSON serialization
        for pattern in patterns.values():
            pattern["endpoints"] = list(pattern["endpoints"])
        
        return sorted(patterns.values(), key=lambda x: x["count"], reverse=True)
    finally:
        db.close()

def get_recent_events(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent telemetry events"""
    db = SessionLocal()
    try:
        events = db.query(TelemetryEvent).order_by(TelemetryEvent.created_at.desc()).limit(limit).all()
        return [
            {
                "evaluation_id": event.evaluation_id,
                "customer_prompt": event.customer_prompt[:100] + "..." if len(event.customer_prompt) > 100 else event.customer_prompt,
                "skill_input": event.skill_input,
                "created_at": event.created_at.isoformat() if event.created_at else "Unknown"
            }
            for event in events
        ]
    finally:
        db.close()

def get_top_clusters(limit: int = 20) -> List[Dict[str, Any]]:
    """Get top clusters by size with full details"""
    db = SessionLocal()
    try:
        from models import ClusterSummary
        # Filter out noise clusters - only show meaningful failure patterns
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).order_by(Cluster.size.desc()).limit(limit).all()
        
        result = []
        for cluster in clusters:
            # Get summary separately to avoid join issues
            summary = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == cluster.id).first()
            
            # Extract business taxonomy from cluster parameters
            taxonomy_category = None
            business_context = {}
            if cluster.cluster_parameters and isinstance(cluster.cluster_parameters, dict):
                taxonomy_category = cluster.cluster_parameters.get('taxonomy_category')
            
            # Map taxonomy to business info
            if taxonomy_category == 'graph_api_issues':
                business_context = {
                    "category_name": "🔍 Graph API Query Issues",
                    "responsible_team": "Graph API Team",
                    "priority": "High",
                    "fix_time": "2-3 weeks",
                    "business_impact": "High - Users can't query data"
                }
            elif taxonomy_category == 'authentication_issues':
                business_context = {
                    "category_name": "🔐 Authentication & Permission Issues",
                    "responsible_team": "Identity Team", 
                    "priority": "Critical",
                    "fix_time": "1-2 days",
                    "business_impact": "Critical - Users can't access resources"
                }
            elif taxonomy_category == 'template_issues':
                business_context = {
                    "category_name": "📝 Template & Variable Issues",
                    "responsible_team": "Copilot Core Team",
                    "priority": "Critical", 
                    "fix_time": "1 week",
                    "business_impact": "Critical - Core functionality broken"
                }
            elif taxonomy_category == 'api_infrastructure':
                business_context = {
                    "category_name": "⚡ API Infrastructure Issues",
                    "responsible_team": "Platform Team",
                    "priority": "High",
                    "fix_time": "3-5 days", 
                    "business_impact": "High - Service reliability affected"
                }
            elif taxonomy_category == 'user_input_issues':
                business_context = {
                    "category_name": "👤 User Input & Intent Issues",
                    "responsible_team": "Copilot UX Team",
                    "priority": "Medium",
                    "fix_time": "2-4 weeks",
                    "business_impact": "Medium - User experience degraded"
                }
            else:
                business_context = {
                    "category_name": f"🔬 Analysis Cluster {cluster.id}",
                    "responsible_team": "Analysis Required",
                    "priority": "Medium",
                    "fix_time": "TBD",
                    "business_impact": "Under investigation"
                }

            # Check for escalation status
            escalation_recommended = cluster.cluster_parameters.get('escalation_recommended', False) if cluster.cluster_parameters else False
            escalation_reasons = cluster.cluster_parameters.get('escalation_reasons', []) if cluster.cluster_parameters else []
            
            cluster_data = {
                "cluster_id": cluster.id,
                "size": cluster.size,
                "algorithm": cluster.cluster_algorithm,
                "taxonomy_category": taxonomy_category,
                "business_context": business_context,
                "summary": summary.summary_text if summary and summary.summary_text else "No AI summary available yet.",
                "root_cause": summary.root_cause if summary and summary.root_cause else "Root cause analysis pending...",
                "recommended_fixes": "Analysis and recommendations are included in the summary above." if summary and summary.summary_text else "Recommended fixes pending...",
                "sample_prompts": summary.sample_prompts if summary and summary.sample_prompts else [],
                "sample_events": [],  # Skip sample events for now to avoid database issues
                "severity": business_context.get("priority", "Medium"),
                "escalation_recommended": escalation_recommended,
                "escalation_reasons": escalation_reasons
            }
            result.append(cluster_data)
        
        return result
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    
    # Get data
    stats = get_system_stats()
    patterns = get_failure_patterns()
    recent_events = get_recent_events()
    clusters = get_top_clusters()
    
    # Create HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Copilot Failure Analysis Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                color: #333;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                border-radius: 20px;
                margin-bottom: 30px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            
            .header h1 {{
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 25px;
                margin-bottom: 40px;
            }}
            
            .stat-card {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border-left: 5px solid #667eea;
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0,0,0,0.15);
            }}
            
            .stat-number {{
                font-size: 3em;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 10px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            }}
            
            .stat-label {{
                color: #666;
                font-size: 1.2em;
                font-weight: 500;
            }}
            
            .section {{
                background: white;
                border-radius: 20px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                overflow: hidden;
            }}
            
            .section-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px 30px;
                font-size: 1.5em;
                font-weight: 600;
            }}
            
            .section-content {{
                padding: 0;
            }}
            
            .cluster-card {{
                border-bottom: 1px solid #eee;
                transition: all 0.3s ease;
                cursor: pointer;
            }}
            
            .cluster-card:last-child {{
                border-bottom: none;
            }}
            
            .cluster-card:hover {{
                background-color: #f8f9ff;
            }}
            
            .cluster-header {{
                padding: 25px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .cluster-info {{
                flex: 1;
            }}
            
            .cluster-title {{
                font-size: 1.3em;
                font-weight: 600;
                color: #333;
                margin-bottom: 8px;
            }}
            
            .cluster-preview {{
                color: #666;
                font-size: 0.95em;
                line-height: 1.4;
                margin-bottom: 10px;
            }}
            
            .cluster-meta {{
                display: flex;
                gap: 15px;
                align-items: center;
            }}
            
            .severity-badge {{
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                text-transform: uppercase;
            }}
            
            .severity-high {{
                background: #ffe6e6;
                color: #d63031;
            }}
            
            .severity-medium {{
                background: #fff5e6;
                color: #e17055;
            }}
            
            .severity-low {{
                background: #e6f7ff;
                color: #0984e3;
            }}
            
            .cluster-size {{
                background: #667eea;
                color: white;
                padding: 8px 16px;
                border-radius: 25px;
                font-weight: 600;
                font-size: 1.1em;
            }}
            
            .expand-icon {{
                font-size: 1.5em;
                color: #667eea;
                transition: transform 0.3s ease;
            }}
            
            .cluster-card.expanded .expand-icon {{
                transform: rotate(180deg);
            }}
            
            .cluster-details {{
                display: none;
                padding: 0 30px 30px 30px;
                background: #f8f9ff;
                border-top: 1px solid #eee;
            }}
            
            .cluster-card.expanded .cluster-details {{
                display: block;
            }}
            
            .detail-section {{
                margin-bottom: 25px;
            }}
            
            .detail-title {{
                font-size: 1.1em;
                font-weight: 600;
                color: #333;
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 2px solid #667eea;
            }}
            
            .detail-content {{
                color: #555;
                line-height: 1.6;
                font-size: 0.95em;
            }}
            
            .sample-events {{
                background: white;
                border-radius: 10px;
                padding: 15px;
                margin-top: 10px;
            }}
            
            .event-item {{
                padding: 12px;
                border-bottom: 1px solid #eee;
                transition: background-color 0.2s;
            }}
            
            .event-item:hover {{
                background-color: #f5f5f5;
            }}
            
            .event-item:last-child {{
                border-bottom: none;
            }}
            
            .event-id {{
                font-weight: 600;
                color: #667eea;
                margin-bottom: 5px;
            }}
            
            .event-prompt {{
                color: #333;
                margin-bottom: 5px;
                font-size: 0.9em;
            }}
            
            .event-time {{
                color: #999;
                font-size: 0.85em;
            }}
            
            .refresh-btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 600;
                margin-bottom: 30px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }}
            
            .refresh-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            }}
            
            .no-data {{
                text-align: center;
                color: #999;
                padding: 60px;
                font-style: italic;
                font-size: 1.1em;
            }}
            
            .search-box {{
                width: 100%;
                padding: 15px 20px;
                border: 2px solid #eee;
                border-radius: 10px;
                font-size: 1em;
                margin-bottom: 20px;
                transition: border-color 0.3s ease;
            }}
            
            .search-box:focus {{
                outline: none;
                border-color: #667eea;
            }}
            
            .filters {{
                display: flex;
                gap: 15px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            
            .filter-btn {{
                padding: 8px 16px;
                border: 2px solid #667eea;
                background: white;
                color: #667eea;
                border-radius: 20px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
            }}
            
            .filter-btn:hover, .filter-btn.active {{
                background: #667eea;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Copilot Failure Analysis Dashboard</h1>
                <p>Real-time insights into Copilot failure patterns and trends</p>
            </div>
            
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{stats['total_events']}</div>
                    <div class="stat-label">Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['total_fingerprints']}</div>
                    <div class="stat-label">Failure Patterns</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['total_embeddings']}</div>
                    <div class="stat-label">Embeddings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['total_clusters']}</div>
                    <div class="stat-label">Clusters</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">🤖 AI-Analyzed Failure Clusters</div>
                <div class="section-content">
                    <div style="padding: 20px 30px;">
                        <input type="text" class="search-box" placeholder="🔍 Search clusters by keywords..." id="searchBox">
                        <div class="filters">
                            <button class="filter-btn active" onclick="filterBySeverity('all')">All Clusters</button>
                            <button class="filter-btn" onclick="filterBySeverity('High')">🔴 High Priority</button>
                            <button class="filter-btn" onclick="filterBySeverity('Medium')">🟡 Medium Priority</button>
                            <button class="filter-btn" onclick="filterBySeverity('Low')">🟢 Low Priority</button>
                        </div>
                    </div>
                    {''.join([
                        f'''
                        <div class="cluster-card" data-severity="{cluster['severity']}" onclick="toggleCluster(this)">
                            <div class="cluster-header">
                                <div class="cluster-info">
                                    <div class="cluster-title">{cluster['business_context']['category_name']}</div>
                                    <div class="cluster-preview">{cluster['summary'][:150] + '...' if len(cluster['summary']) > 150 else cluster['summary']}</div>
                                    <div class="cluster-meta">
                                        <span class="severity-badge severity-{cluster['severity'].lower()}">{cluster['severity']} Priority</span>
                                        <span style="color: #666; font-size: 0.9em;">Team: {cluster['business_context']['responsible_team']}</span>
                                        <span style="color: #666; font-size: 0.9em;">Fix Time: {cluster['business_context']['fix_time']}</span>
                                    </div>
                                </div>
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <div class="cluster-size">{cluster['size']}</div>
                                    <div class="expand-icon">▼</div>
                                </div>
                            </div>
                            {f'''
                            <div class="escalation-alert" style="background: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 10px; border-radius: 5px;">
                                <div style="color: #d32f2f; font-weight: bold; font-size: 1.1em; margin-bottom: 8px;">🚨 ESCALATION RECOMMENDED</div>
                                <div style="color: #d32f2f; font-size: 0.95em;">
                                    <strong>Reasons:</strong> {'; '.join(cluster['escalation_reasons'])}<br>
                                    <strong>Action Required:</strong> Immediate team notification and priority triage<br>
                                    <strong>Next Steps:</strong> Contact {cluster['business_context']['responsible_team']} within 24 hours
                                </div>
                            </div>
                            ''' if cluster['escalation_recommended'] else ''}
                            <div class="cluster-details">
                                <div class="detail-section">
                                    <div class="detail-title">🏢 Business Context</div>
                                    <div class="detail-content">
                                        <strong>Responsible Team:</strong> {cluster['business_context']['responsible_team']}<br>
                                        <strong>Priority:</strong> {cluster['business_context']['priority']}<br>
                                        <strong>Typical Fix Time:</strong> {cluster['business_context']['fix_time']}<br>
                                        <strong>Business Impact:</strong> {cluster['business_context']['business_impact']}
                                    </div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">📋 AI Summary</div>
                                    <div class="detail-content">{cluster['summary']}</div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">🔍 Root Cause Analysis</div>
                                    <div class="detail-content">{cluster['root_cause']}</div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">🛠️ Recommended Fixes</div>
                                    <div class="detail-content">{cluster['recommended_fixes']}</div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">📝 Sample Events ({len(cluster['sample_events'])} shown)</div>
                                    <div class="sample-events">
                                        {''.join([
                                            f'''
                                            <div class="event-item">
                                                <div class="event-id">{event['evaluation_id']}</div>
                                                <div class="event-prompt">{event['customer_prompt']}</div>
                                                <div class="event-time">Skill: {event['skill_input']} | {event['created_at']}</div>
                                            </div>
                                            '''
                                            for event in cluster['sample_events']
                                        ])}
                                    </div>
                                </div>
                            </div>
                        </div>
                        '''
                        for cluster in clusters
                    ]) if clusters else '<div class="no-data">No clusters found. Try running the analysis first.</div>'}
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">🕒 Recent Events</div>
                <div class="section-content">
                    {''.join([
                        f'''
                        <div class="event-item">
                            <div class="event-id">{event['evaluation_id']}</div>
                            <div class="event-prompt">{event['customer_prompt']}</div>
                            <div class="event-time">Skill: {event['skill_input']} | {event['created_at']}</div>
                        </div>
                        '''
                        for event in recent_events
                    ]) if recent_events else '<div class="no-data">No recent events found. Try ingesting some data first.</div>'}
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">🔗 System Information</div>
                <div class="section-content" style="padding: 30px;">
                    <p style="margin-bottom: 20px; color: #666; font-size: 1.1em;">System analysis status and quick actions:</p>
                    <ul style="list-style: none; padding: 0;">
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Total Events:</strong> {stats['total_events']} telemetry events processed</li>
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Embeddings:</strong> {stats['total_embeddings']} vector embeddings generated</li>
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Clusters:</strong> {stats['total_clusters']} failure clusters identified</li>
                        <li style="padding: 10px 0;"><a href="/docs" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 600;">📚 API Documentation</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">🚀 Quick Actions</div>
                <div class="section-content" style="padding: 30px;">
                    <p style="margin-bottom: 20px; color: #666; font-size: 1.1em;">Commands to manage and analyze your data:</p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                        <div style="background: #f8f9ff; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;">
                            <strong style="color: #667eea;">Run Full Analysis</strong>
                            <div style="font-family: monospace; background: #333; color: #fff; padding: 10px; border-radius: 5px; margin-top: 10px;">python main.py</div>
                        </div>
                        <div style="background: #f8f9ff; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;">
                            <strong style="color: #667eea;">Import Excel Data</strong>
                            <div style="font-family: monospace; background: #333; color: #fff; padding: 10px; border-radius: 5px; margin-top: 10px;">python excel_import.py import file.xlsx</div>
                        </div>
                        <div style="background: #f8f9ff; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;">
                            <strong style="color: #667eea;">Generate AI Summaries</strong>
                            <div style="font-family: monospace; background: #333; color: #fff; padding: 10px; border-radius: 5px; margin-top: 10px;">python generate_summaries.py</div>
                        </div>
                        <div style="background: #f8f9ff; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;">
                            <strong style="color: #667eea;">Refresh Dashboard</strong>
                            <div style="font-family: monospace; background: #333; color: #fff; padding: 10px; border-radius: 5px; margin-top: 10px;">Ctrl+R or click Refresh</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function toggleCluster(element) {{
                element.classList.toggle('expanded');
            }}
            
            function filterBySeverity(severity) {{
                // Update active filter button
                document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');
                
                // Show/hide clusters
                document.querySelectorAll('.cluster-card').forEach(card => {{
                    if (severity === 'all' || card.dataset.severity === severity) {{
                        card.style.display = 'block';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }});
            }}
            
            // Search functionality
            document.getElementById('searchBox').addEventListener('input', function() {{
                const searchTerm = this.value.toLowerCase();
                document.querySelectorAll('.cluster-card').forEach(card => {{
                    const content = card.textContent.toLowerCase();
                    if (content.includes(searchTerm)) {{
                        card.style.display = 'block';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }});
            }});
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'r' && e.ctrlKey) {{
                    e.preventDefault();
                    location.reload();
                }}
                if (e.key === 'Escape') {{
                    document.getElementById('searchBox').value = '';
                    document.getElementById('searchBox').dispatchEvent(new Event('input'));
                }}
            }});
            
            // Auto-refresh every 60 seconds (increased from 30)
            setTimeout(() => location.reload(), 60000);
            
            // Add loading animation on refresh
            window.addEventListener('beforeunload', function() {{
                document.body.style.opacity = '0.7';
                document.body.style.transition = 'opacity 0.3s ease';
            }});
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/api/stats")
async def api_stats():
    """API endpoint for statistics"""
    return {
        "stats": get_system_stats(),
        "patterns": get_failure_patterns(),
        "recent_events": get_recent_events()
    }

if __name__ == "__main__":
    print("🚀 Starting Copilot Failure Analysis Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8080")
    print("📚 API documentation at: http://localhost:8080/docs")
    print("🔄 Auto-refreshes every 30 seconds")
    print("\n" + "="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8080) 