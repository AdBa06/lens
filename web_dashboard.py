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
from collections import Counter
import re

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

def get_cluster_events(db, cluster_id: int) -> List[Dict[str, Any]]:
    """Get ALL events for a specific cluster with their customer prompts"""
    try:
        from models import ClusterAssignment, Embedding
        
        # Get all events in this cluster
        events = db.query(TelemetryEvent)\
                   .join(Embedding, TelemetryEvent.id == Embedding.event_id)\
                   .join(ClusterAssignment, Embedding.id == ClusterAssignment.embedding_id)\
                   .filter(ClusterAssignment.cluster_id == cluster_id)\
                   .all()
        
        result = []
        for event in events:
            result.append({
                "evaluation_id": event.evaluation_id,
                "customer_prompt": event.customer_prompt or "No prompt available",
                "created_at": event.created_at.strftime("%Y-%m-%d %H:%M") if event.created_at else "Unknown"
            })
        
        return result
    except Exception as e:
        print(f"Error getting cluster events: {e}")
        return []

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
                    "category_name": "Graph API Query Issues",
                    "responsible_team": "Graph API Team",
                    "priority": "High",
                    "fix_time": "2-3 weeks",
                    "business_impact": "High - Users can't query data"
                }
            elif taxonomy_category == 'authentication_issues':
                business_context = {
                    "category_name": "Authentication & Permission Issues",
                    "responsible_team": "Identity Team", 
                    "priority": "Critical",
                    "fix_time": "1-2 days",
                    "business_impact": "Critical - Users can't access resources"
                }
            elif taxonomy_category == 'template_issues':
                business_context = {
                    "category_name": "Template & Variable Issues",
                    "responsible_team": "Copilot Core Team",
                    "priority": "Critical", 
                    "fix_time": "1 week",
                    "business_impact": "Critical - Core functionality broken"
                }
            elif taxonomy_category == 'api_infrastructure':
                business_context = {
                    "category_name": "API Infrastructure Issues",
                    "responsible_team": "Platform Team",
                    "priority": "High",
                    "fix_time": "3-5 days", 
                    "business_impact": "High - Service reliability affected"
                }
            elif taxonomy_category == 'user_input_issues':
                business_context = {
                    "category_name": "User Input & Intent Issues",
                    "responsible_team": "Copilot UX Team",
                    "priority": "Medium",
                    "fix_time": "2-4 weeks",
                    "business_impact": "Medium - User experience degraded"
                }
            else:
                business_context = {
                    "category_name": f"Analysis Cluster {cluster.id}",
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
                "all_events": get_cluster_events(db, cluster.id),  # Get ALL events in this cluster
                "severity": business_context.get("priority", "Medium"),
                "escalation_recommended": escalation_recommended,
                "escalation_reasons": escalation_reasons
            }
            result.append(cluster_data)
        
        return result
    finally:
        db.close()

def get_prompt_analytics() -> Dict[str, Any]:
    """Analyze customer prompts for patterns and insights"""
    db = SessionLocal()
    try:
        # Get all events
        events = db.query(TelemetryEvent).all()
        
        # Extract prompts
        prompts = [event.customer_prompt for event in events if event.customer_prompt]
        
        # Analyze prompt characteristics
        prompt_lengths = [len(prompt) for prompt in prompts]
        
        # Categorize by intent (simple keyword matching)
        categories = {
            "Calendar & Meetings": ["calendar", "meeting", "schedule appointment", "book meeting", "create event"],
            "Email & Messages": ["email", "message", "send", "reply", "inbox", "mail"],
            "Teams & Collaboration": ["teams", "chat", "share", "collaborate", "call", "video"],
            "Data & Reports": ["report", "data", "chart", "graph", "dashboard", "analysis"],
            "Files & Documents": ["file", "document", "folder", "upload", "download", "open"],
            "Search & Query": ["find", "search", "look", "query", "get", "show"],
            "Settings & Config": ["setting", "config", "setup", "preference", "enable", "disable"],
            "Help & Support": ["help", "how", "what", "why", "support", "issue"]
        }
        
        category_counts = {}
        for category, keywords in categories.items():
            count = sum(1 for prompt in prompts 
                       if any(keyword.lower() in prompt.lower() for keyword in keywords))
            category_counts[category] = count
        
        # Common words analysis
        all_words = []
        for prompt in prompts:
            words = re.findall(r'\b\w+\b', prompt.lower())
            all_words.extend([word for word in words if len(word) > 3])  # Filter short words
        
        common_words = dict(Counter(all_words).most_common(20))
        
        # Prompt complexity analysis
        complexity_levels = {
            "Simple (< 50 chars)": len([p for p in prompts if len(p) < 50]),
            "Medium (50-150 chars)": len([p for p in prompts if 50 <= len(p) <= 150]),
            "Complex (> 150 chars)": len([p for p in prompts if len(p) > 150])
        }
        
        return {
            "total_prompts": len(prompts),
            "avg_length": round(sum(prompt_lengths) / len(prompt_lengths)) if prompt_lengths else 0,
            "categories": category_counts,
            "common_words": common_words,
            "complexity": complexity_levels,
            "sample_prompts": prompts[:10]  # First 10 for display
        }
        
    finally:
        db.close()

def get_error_analytics() -> Dict[str, Any]:
    """Deep analysis of error patterns and distributions"""
    db = SessionLocal()
    try:
        # Get all fingerprints and events
        fingerprints = db.query(FailureFingerprint).all()
        events = db.query(TelemetryEvent).all()
        
        # Error type distribution
        error_types = [fp.error_type for fp in fingerprints if fp.error_type]
        error_type_counts = dict(Counter(error_types).most_common(10))
        
        # Plugin distribution
        plugins = [fp.plugin_name for fp in fingerprints if fp.plugin_name]
        plugin_counts = dict(Counter(plugins).most_common(10))
        
        # Status code distribution
        status_codes = [str(fp.status_code) for fp in fingerprints if fp.status_code]
        status_code_counts = dict(Counter(status_codes).most_common(10))
        
        # Endpoint analysis
        endpoints = [fp.endpoint for fp in fingerprints if fp.endpoint]
        endpoint_counts = dict(Counter(endpoints).most_common(10))
        
        # Error message patterns
        error_messages = [fp.error_message for fp in fingerprints if fp.error_message]
        
        # Critical error analysis (4xx and 5xx codes)
        critical_errors = [fp for fp in fingerprints if fp.status_code and fp.status_code >= 400]
        critical_count = len(critical_errors)
        
        # Most problematic combinations
        plugin_error_combos = {}
        for fp in fingerprints:
            if fp.plugin_name and fp.error_type:
                combo = f"{fp.plugin_name}/{fp.error_type}"
                plugin_error_combos[combo] = plugin_error_combos.get(combo, 0) + 1
        
        top_combos = dict(Counter(plugin_error_combos).most_common(5))
        
        return {
            "total_errors": len(fingerprints),
            "critical_errors": critical_count,
            "error_types": error_type_counts,
            "plugins": plugin_counts,
            "status_codes": status_code_counts,
            "endpoints": endpoint_counts,
            "top_combinations": top_combos,
            "sample_messages": error_messages[:10]
        }
        
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
            
            .nav-bar {{
                background: rgba(255,255,255,0.2);
                border-radius: 15px;
                padding: 15px;
                margin-top: 20px;
                display: flex;
                justify-content: center;
                gap: 20px;
                flex-wrap: wrap;
            }}
            
            .nav-btn {{
                background: rgba(255,255,255,0.9);
                color: #333;
                padding: 12px 24px;
                border-radius: 25px;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .nav-btn:hover {{
                background: white;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
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
                <h1>Copilot Failure Analysis Dashboard</h1>
                <p>Real-time insights into Copilot failure patterns and trends</p>
            </div>
            
            <div style="background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center;">
                <div>
                                          <a href="/" style="display: inline-block; padding: 12px 25px; margin: 0 10px; text-decoration: none; border-radius: 25px; font-weight: 600; transition: all 0.3s ease; background: #667eea; color: white;">Main Dashboard</a>
                    <a href="/prompts" style="display: inline-block; padding: 12px 25px; margin: 0 10px; text-decoration: none; border-radius: 25px; font-weight: 600; transition: all 0.3s ease; background: #f8f9ff; color: #667eea;">Prompt Analytics</a>
                    <a href="/errors" style="display: inline-block; padding: 12px 25px; margin: 0 10px; text-decoration: none; border-radius: 25px; font-weight: 600; transition: all 0.3s ease; background: #f8f9ff; color: #667eea;">Error Analytics</a>
                </div>
                <button class="refresh-btn" onclick="location.reload()" style="background: #667eea; color: white; border: none; padding: 12px 20px; border-radius: 25px; cursor: pointer; font-weight: 600; transition: all 0.3s ease;">Refresh Data</button>
            </div>
            
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
                <div class="section-header">AI-Analyzed Failure Clusters</div>
                <div class="section-content">
                    <div style="padding: 20px 30px;">
                        <input type="text" class="search-box" placeholder="Search clusters by keywords..." id="searchBox">
                        <div class="filters">
                            <button class="filter-btn active" onclick="filterBySeverity('all')">All Clusters</button>
                            <button class="filter-btn" onclick="filterBySeverity('High')">High Priority</button>
                            <button class="filter-btn" onclick="filterBySeverity('Medium')">Medium Priority</button>
                            <button class="filter-btn" onclick="filterBySeverity('Low')">Low Priority</button>
                        </div>
                    </div>
                    {''.join([
                        f'''
                        <div class="cluster-card" data-severity="{cluster['severity']}" onclick="toggleCluster(this)">
                            <div class="cluster-header" style="position: relative;">
                                <div class="cluster-info" style="flex: 1; padding-right: 20px;">
                                    <div class="cluster-title">{cluster['business_context']['category_name']}</div>
                                    <div class="cluster-preview">{cluster['summary'][:150] + '...' if len(cluster['summary']) > 150 else cluster['summary']}</div>
                                    <div class="cluster-meta">
                                        <span class="severity-badge severity-{cluster['severity'].lower()}">{cluster['severity']} Priority</span>
                                        <span style="color: #666; font-size: 0.9em;">Team: {cluster['business_context']['responsible_team']}</span>
                                        <span style="color: #666; font-size: 0.9em;">Fix Time: {cluster['business_context']['fix_time']}</span>
                                    </div>
                                </div>
                                {f'''
                                <div class="escalation-box" style="position: absolute; right: 120px; top: 50%; transform: translateY(-50%); background: #ffebee; border: 2px solid #f44336; padding: 12px; border-radius: 8px; max-width: 250px; box-shadow: 0 4px 12px rgba(244, 67, 54, 0.2);">
                                    <div style="color: #d32f2f; font-weight: bold; font-size: 0.9em; margin-bottom: 5px;">ESCALATION REQUIRED</div>
                                    <div style="color: #d32f2f; font-size: 0.8em; line-height: 1.3;">
                                        Critical business function impacted<br>
                                        <strong>Contact {cluster['business_context']['responsible_team']} within 24h</strong>
                                    </div>
                                </div>
                                ''' if cluster['escalation_recommended'] else ''}
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <div class="cluster-size">{cluster['size']}</div>
                                    <div class="expand-icon">▼</div>
                                </div>
                            </div>
                            <div class="cluster-details">
                                <div class="detail-section">
                                    <div class="detail-title">Business Context</div>
                                    <div class="detail-content">
                                        <strong>Responsible Team:</strong> {cluster['business_context']['responsible_team']}<br>
                                        <strong>Priority:</strong> {cluster['business_context']['priority']}<br>
                                        <strong>Typical Fix Time:</strong> {cluster['business_context']['fix_time']}<br>
                                        <strong>Business Impact:</strong> {cluster['business_context']['business_impact']}
                                    </div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">AI Summary</div>
                                    <div class="detail-content">{cluster['summary']}</div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">Root Cause Analysis</div>
                                    <div class="detail-content">{cluster['root_cause']}</div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">Recommended Fixes</div>
                                    <div class="detail-content">{cluster['recommended_fixes']}</div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">Sample Events ({len(cluster.get('all_events', []))} shown)</div>
                                    <div class="sample-events">
                                        {''.join([
                                            f'''
                                            <div class="event-item">
                                                <div class="event-id">{event.get('evaluation_id', 'N/A')}</div>
                                                <div class="event-prompt">{event.get('customer_prompt', 'No prompt available')}</div>
                                                <div class="event-time">Event: {event.get('created_at', 'Unknown time')}</div>
                                            </div>
                                            '''
                                            for event in cluster.get('all_events', [])
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
                <div class="section-header">Recent Events</div>
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
                <div class="section-header">System Information</div>
                <div class="section-content" style="padding: 30px;">
                    <p style="margin-bottom: 20px; color: #666; font-size: 1.1em;">System analysis status and quick actions:</p>
                    <ul style="list-style: none; padding: 0;">
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Total Events:</strong> {stats['total_events']} telemetry events processed</li>
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Embeddings:</strong> {stats['total_embeddings']} vector embeddings generated</li>
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Clusters:</strong> {stats['total_clusters']} failure clusters identified</li>
                        <li style="padding: 10px 0;"><a href="/docs" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 600;">API Documentation</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">Quick Actions</div>
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

@app.get("/prompts", response_class=HTMLResponse)
async def prompt_analytics():
    """Prompt Analytics Dashboard"""
    data = get_prompt_analytics()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prompt Analytics - Copilot Analysis</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh; color: #333;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            .header {{ 
                background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%);
                color: white; padding: 40px; border-radius: 20px; margin-bottom: 30px;
                text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .header h1 {{ font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
            .nav {{ background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); }}
            .nav a {{ 
                display: inline-block; padding: 12px 25px; margin: 0 10px; text-decoration: none;
                border-radius: 25px; font-weight: 600; transition: all 0.3s ease;
            }}
            .nav a.active {{ background: #42a5f5; color: white; }}
            .nav a:not(.active) {{ background: #f8f9ff; color: #42a5f5; }}
            .nav a:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }}
            .stats-grid {{ 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 25px; margin-bottom: 40px;
            }}
            .stat-card {{ 
                background: white; padding: 30px; border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center;
                transition: transform 0.3s ease; border-left: 5px solid #42a5f5;
            }}
            .stat-card:hover {{ transform: translateY(-5px); }}
            .stat-number {{ font-size: 3em; font-weight: 700; color: #42a5f5; margin-bottom: 10px; }}
            .stat-label {{ color: #666; font-size: 1.2em; font-weight: 500; }}
            .section {{ 
                background: white; border-radius: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                margin-bottom: 30px; overflow: hidden;
            }}
            .section-header {{ 
                background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%);
                color: white; padding: 25px 30px; font-size: 1.5em; font-weight: 600;
            }}
            .section-content {{ padding: 30px; }}
            .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; }}
            .chart-item {{ background: #f8f9ff; padding: 20px; border-radius: 10px; border-left: 4px solid #42a5f5; }}
            .chart-title {{ font-weight: 600; color: #42a5f5; margin-bottom: 15px; font-size: 1.1em; }}
            .bar-chart {{ margin: 10px 0; }}
            .bar-item {{ display: flex; align-items: center; margin: 8px 0; }}
            .bar-label {{ min-width: 150px; font-size: 0.9em; }}
            .bar-visual {{ 
                flex: 1; height: 20px; background: #e3f2fd; border-radius: 10px; margin: 0 10px;
                position: relative; overflow: hidden;
            }}
            .bar-fill {{ 
                height: 100%; background: linear-gradient(90deg, #42a5f5, #1e88e5); border-radius: 10px;
                transition: width 0.8s ease;
            }}
            .bar-value {{ min-width: 40px; font-weight: 600; color: #42a5f5; }}
            .word-cloud {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }}
            .word-tag {{ 
                background: #42a5f5; color: white; padding: 8px 16px; border-radius: 20px;
                font-size: 0.9em; font-weight: 500;
            }}
            .sample-list {{ background: #f8f9ff; padding: 20px; border-radius: 10px; margin-top: 20px; }}
            .sample-item {{ padding: 10px; border-bottom: 1px solid #e0e0e0; }}
            .sample-item:last-child {{ border-bottom: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Customer Prompt Analytics</h1>
                <p>Understanding what customers ask and how they interact with Copilot</p>
            </div>
            
            <div class="nav">
                <a href="/">Main Dashboard</a>
                <a href="/prompts" class="active">Prompt Analytics</a>
                <a href="/errors">Error Analytics</a>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{data['total_prompts']}</div>
                    <div class="stat-label">Total Customer Prompts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{data['avg_length']}</div>
                    <div class="stat-label">Average Prompt Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(data['categories'])}</div>
                    <div class="stat-label">Intent Categories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(data['common_words'])}</div>
                    <div class="stat-label">Unique Keywords</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">Prompt Categories by Intent</div>
                <div class="section-content">
                    <div class="chart-grid">
                        <div class="chart-item">
                            <div class="chart-title">Customer Intent Distribution</div>
                            <div class="bar-chart">
                                {'\n'.join([
                                    f'''
                                    <div class="bar-item">
                                        <div class="bar-label">{category}</div>
                                        <div class="bar-visual">
                                            <div class="bar-fill" style="width: {(count / max(data['categories'].values()) * 100) if data['categories'].values() else 0}%"></div>
                                        </div>
                                        <div class="bar-value">{count}</div>
                                    </div>
                                    '''
                                    for category, count in data['categories'].items()
                                ])}
                            </div>
                        </div>
                        
                        <div class="chart-item">
                            <div class="chart-title">Prompt Complexity Levels</div>
                            <div class="bar-chart">
                                {'\n'.join([
                                    f'''
                                    <div class="bar-item">
                                        <div class="bar-label">{level}</div>
                                        <div class="bar-visual">
                                            <div class="bar-fill" style="width: {(count / max(data['complexity'].values()) * 100) if data['complexity'].values() else 0}%"></div>
                                        </div>
                                        <div class="bar-value">{count}</div>
                                    </div>
                                    '''
                                    for level, count in data['complexity'].items()
                                ])}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">🔤 Most Common Keywords</div>
                <div class="section-content">
                    <p style="margin-bottom: 20px; color: #666;">Keywords that appear most frequently in customer prompts (filtered for meaningful words)</p>
                    <div class="word-cloud">
                        {'\n'.join([
                            f'<span class="word-tag" style="font-size: {0.8 + (count / max(data["common_words"].values()) * 0.6)}em;">{word} ({count})</span>'
                            for word, count in list(data['common_words'].items())[:15]
                        ])}
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">Sample Customer Prompts</div>
                <div class="section-content">
                    <p style="margin-bottom: 20px; color: #666;">Representative examples of what customers are asking</p>
                    <div class="sample-list">
                        {'\n'.join([
                            f'<div class="sample-item">"{prompt[:200]}{"..." if len(prompt) > 200 else ""}"</div>'
                            for prompt in data['sample_prompts']
                        ])}
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Animate bars on page load
            window.addEventListener('load', function() {{
                const bars = document.querySelectorAll('.bar-fill');
                bars.forEach(bar => {{
                    const width = bar.style.width;
                    bar.style.width = '0%';
                    setTimeout(() => bar.style.width = width, 100);
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/errors", response_class=HTMLResponse)
async def error_analytics():
    """Clustered Error Analytics Dashboard"""
    clusters = get_top_clusters()
    data = get_error_analytics()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error Analytics - Copilot Analysis</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh; color: #333;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            .header {{ 
                background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
                color: white; padding: 40px; border-radius: 20px; margin-bottom: 30px;
                text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .header h1 {{ font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
            .nav {{ background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); }}
            .nav a {{ 
                display: inline-block; padding: 12px 25px; margin: 0 10px; text-decoration: none;
                border-radius: 25px; font-weight: 600; transition: all 0.3s ease;
            }}
            .nav a.active {{ background: #f44336; color: white; }}
            .nav a:not(.active) {{ background: #f8f9ff; color: #f44336; }}
            .nav a:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }}
            .stats-grid {{ 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 25px; margin-bottom: 40px;
            }}
            .stat-card {{ 
                background: white; padding: 30px; border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center;
                transition: transform 0.3s ease; border-left: 5px solid #f44336;
            }}
            .stat-card:hover {{ transform: translateY(-5px); }}
            .stat-number {{ font-size: 3em; font-weight: 700; color: #f44336; margin-bottom: 10px; }}
            .stat-label {{ color: #666; font-size: 1.2em; font-weight: 500; }}
            .section {{ 
                background: white; border-radius: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                margin-bottom: 30px; overflow: hidden;
            }}
            .section-header {{ 
                background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
                color: white; padding: 25px 30px; font-size: 1.5em; font-weight: 600;
            }}
            .section-content {{ padding: 30px; }}
            .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; }}
            .chart-item {{ background: #fff5f5; padding: 20px; border-radius: 10px; border-left: 4px solid #f44336; }}
            .chart-title {{ font-weight: 600; color: #f44336; margin-bottom: 15px; font-size: 1.1em; }}
            .bar-chart {{ margin: 10px 0; }}
            .bar-item {{ display: flex; align-items: center; margin: 8px 0; }}
            .bar-label {{ min-width: 150px; font-size: 0.9em; }}
            .bar-visual {{ 
                flex: 1; height: 20px; background: #ffebee; border-radius: 10px; margin: 0 10px;
                position: relative; overflow: hidden;
            }}
            .bar-fill {{ 
                height: 100%; background: linear-gradient(90deg, #f44336, #d32f2f); border-radius: 10px;
                transition: width 0.8s ease;
            }}
            .bar-value {{ min-width: 40px; font-weight: 600; color: #f44336; }}
            .critical-alert {{ 
                background: #ffebee; border: 2px solid #f44336; border-radius: 10px; padding: 20px;
                margin-bottom: 30px; text-align: center;
            }}
            .critical-number {{ font-size: 2.5em; font-weight: 700; color: #f44336; }}
            .sample-list {{ background: #fff5f5; padding: 20px; border-radius: 10px; margin-top: 20px; }}
            .sample-item {{ padding: 10px; border-bottom: 1px solid #e0e0e0; font-family: monospace; }}
            .sample-item:last-child {{ border-bottom: none; }}
            
            /* Expandable cluster details */
            .cluster-details {{ margin-top: 15px; }}
            .cluster-details details {{ 
                background: white; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 10px;
            }}
            .cluster-details summary {{ 
                padding: 15px; font-weight: 600; color: #f44336; cursor: pointer; 
                background: #fafafa; border-radius: 8px; list-style: none;
                transition: background 0.3s ease;
            }}
            .cluster-details summary:hover {{ background: #f0f0f0; }}
            .cluster-details summary::-webkit-details-marker {{ display: none; }}
            .cluster-details summary::before {{ 
                content: "▶️ "; margin-right: 8px; transition: transform 0.3s ease;
            }}
            .cluster-details details[open] summary::before {{ transform: rotate(90deg); }}
            
            .query-list {{ max-height: 400px; overflow-y: auto; }}
            .query-item {{ 
                padding: 12px; border-bottom: 1px solid #eee; font-size: 0.9em; line-height: 1.4;
                transition: background 0.2s ease;
            }}
            .query-item:hover {{ background: #f8f9ff; }}
            .query-item:last-child {{ border-bottom: none; }}
            .query-id {{ color: #f44336; font-weight: 600; font-size: 0.85em; }}
            .query-text {{ color: #333; margin-top: 5px; }}
            .query-time {{ color: #999; font-size: 0.8em; margin-top: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Error Pattern Analytics</h1>
                <p>Deep dive analysis of failure patterns and system errors</p>
            </div>
            
            <div class="nav">
                <a href="/">Main Dashboard</a>
                <a href="/prompts">Prompt Analytics</a>
                <a href="/errors" class="active">Error Analytics</a>
            </div>
            
            <div class="critical-alert">
                <div class="critical-number">{len(clusters)}</div>
                <div style="color: #f44336; font-weight: 600; font-size: 1.2em;">Error Clusters Identified</div>
                <div style="color: #666; margin-top: 10px;">Groups of similar failures for targeted analysis</div>
            </div>
            
            <div class="section">
                <div class="section-header">🎯 Error Clusters - Similar Failure Patterns</div>
                <div class="section-content">
                    {''.join([
                        f'''
                        <div style="background: #fff5f5; border: 1px solid #ffcdd2; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h3 style="color: #f44336; margin: 0;">🎯 Cluster {cluster['cluster_id']} - {cluster['size']} Similar Errors</h3>
                                <span style="background: #f44336; color: white; padding: 5px 15px; border-radius: 20px; font-weight: 600;">{cluster['algorithm']}</span>
                            </div>
                            <div style="color: #666; margin-bottom: 15px; line-height: 1.6;">
                                <strong>Pattern Summary:</strong> {cluster['summary'] if cluster['summary'] else 'Analysis in progress...'}
                            </div>
                            {f'<div style="color: #666; margin-bottom: 15px;"><strong>Root Cause:</strong> {cluster["root_cause"]}</div>' if cluster.get('root_cause') else ''}
                            
                            <div class="cluster-details">
                                <details>
                                    <summary>View All {cluster['size']} Customer Queries in this Cluster</summary>
                                    <div class="query-list">
                                        {''.join([
                                            f'''
                                            <div class="query-item">
                                                <div class="query-id">Event ID: {event["evaluation_id"]}</div>
                                                <div class="query-text">{event["customer_prompt"]}</div>
                                                <div class="query-time">Timestamp: {event["created_at"]}</div>
                                            </div>
                                            '''
                                            for event in cluster['all_events']
                                        ]) if cluster.get('all_events') else '<div style="padding: 15px; color: #999; text-align: center;">No queries available for this cluster</div>'}
                                    </div>
                                </details>
                            </div>
                        </div>
                        '''
                        for cluster in clusters
                    ]) if clusters else '<div style="text-align: center; color: #666; padding: 40px;">No error clusters found. Run clustering analysis first.</div>'}
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{data['total_errors']}</div>
                    <div class="stat-label">Total Error Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(data['error_types'])}</div>
                    <div class="stat-label">Unique Error Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(data['plugins'])}</div>
                    <div class="stat-label">Affected Plugins</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(data['endpoints'])}</div>
                    <div class="stat-label">Problem Endpoints</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">Error Distribution Analysis</div>
                <div class="section-content">
                    <div class="chart-grid">
                        <div class="chart-item">
                            <div class="chart-title">Most Common Error Types</div>
                            <div class="bar-chart">
                                {'\n'.join([
                                    f'''
                                    <div class="bar-item">
                                        <div class="bar-label">{error_type}</div>
                                        <div class="bar-visual">
                                            <div class="bar-fill" style="width: {(count / max(data['error_types'].values()) * 100) if data['error_types'].values() else 0}%"></div>
                                        </div>
                                        <div class="bar-value">{count}</div>
                                    </div>
                                    '''
                                    for error_type, count in data['error_types'].items()
                                ])}
                            </div>
                        </div>
                        
                        <div class="chart-item">
                            <div class="chart-title">HTTP Status Code Distribution</div>
                            <div class="bar-chart">
                                {'\n'.join([
                                    f'''
                                    <div class="bar-item">
                                        <div class="bar-label">Status {status}</div>
                                        <div class="bar-visual">
                                            <div class="bar-fill" style="width: {(count / max(data['status_codes'].values()) * 100) if data['status_codes'].values() else 0}%"></div>
                                        </div>
                                        <div class="bar-value">{count}</div>
                                    </div>
                                    '''
                                    for status, count in data['status_codes'].items()
                                ])}
                            </div>
                        </div>
                        
                        <div class="chart-item">
                            <div class="chart-title">Plugin Error Frequency</div>
                            <div class="bar-chart">
                                {'\n'.join([
                                    f'''
                                    <div class="bar-item">
                                        <div class="bar-label">{plugin}</div>
                                        <div class="bar-visual">
                                            <div class="bar-fill" style="width: {(count / max(data['plugins'].values()) * 100) if data['plugins'].values() else 0}%"></div>
                                        </div>
                                        <div class="bar-value">{count}</div>
                                    </div>
                                    '''
                                    for plugin, count in data['plugins'].items()
                                ])}
                            </div>
                        </div>
                        
                        <div class="chart-item">
                            <div class="chart-title">Most Problematic Plugin/Error Combinations</div>
                            <div class="bar-chart">
                                {'\n'.join([
                                    f'''
                                    <div class="bar-item">
                                        <div class="bar-label">{combo}</div>
                                        <div class="bar-visual">
                                            <div class="bar-fill" style="width: {(count / max(data['top_combinations'].values()) * 100) if data['top_combinations'].values() else 0}%"></div>
                                        </div>
                                        <div class="bar-value">{count}</div>
                                    </div>
                                    '''
                                    for combo, count in data['top_combinations'].items()
                                ])}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">Sample Error Messages</div>
                <div class="section-content">
                    <p style="margin-bottom: 20px; color: #666;">Raw error messages from the system for debugging reference</p>
                    <div class="sample-list">
                        {'\n'.join([
                            f'<div class="sample-item">{message[:200] + "..." if len(message) > 200 else message}</div>'
                            for message in data['sample_messages'] if message
                        ])}
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Animate bars on page load
            window.addEventListener('load', function() {{
                const bars = document.querySelectorAll('.bar-fill');
                bars.forEach(bar => {{
                    const width = bar.style.width;
                    bar.style.width = '0%';
                    setTimeout(() => bar.style.width = width, 100);
                }});
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
    print("Starting Copilot Failure Analysis Dashboard...")
    print("Dashboard will be available at: http://localhost:8080")
    print("API documentation at: http://localhost:8080/docs")
    print("Auto-refreshes every 30 seconds")
    print("\n" + "="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8080) 