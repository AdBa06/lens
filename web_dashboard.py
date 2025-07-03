#!/usr/bin/env python3
"""
Simple 2-Page Dashboard for Leadership
1. Main Analytics: Customer prompt patterns and insights 
2. Cluster Analytics: Failure clusters for leadership review
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
from typing import Dict, List, Any
from database import SessionLocal
from models import TelemetryEvent, Embedding, Cluster, ClusterSummary, ClusterAssignment
import uvicorn
from collections import Counter
import re

app = FastAPI(title="Security Investigation Analytics Dashboard")

def get_prompt_analytics() -> Dict[str, Any]:
    """Analyze customer prompts for leadership insights"""
    db = SessionLocal()
    try:
        events = db.query(TelemetryEvent).all()
        
        # Extract prompt themes
        themes = {
            "signin_investigation": [],
            "group_membership": [],
            "audit_logs": [],
            "suspicious_activity": [],
            "defender_queries": [],
            "sentinel_investigation": [],
            "entra_id_analysis": [],
            "other": []
        }
        
        # Security tools mentioned
        tools_mentioned = Counter()
        
        # Investigation types
        investigation_types = Counter()
        
        for event in events:
            prompt = event.customer_prompt.lower() if event.customer_prompt is not None else ""
            
            # Categorize by theme
            if any(word in prompt for word in ["signin", "sign-in", "login"]):
                themes["signin_investigation"].append(event.customer_prompt[:200] + "...")
                investigation_types["Signin Analysis"] += 1
            elif any(word in prompt for word in ["group", "membership", "role"]):
                themes["group_membership"].append(event.customer_prompt[:200] + "...")
                investigation_types["Access Management"] += 1
            elif any(word in prompt for word in ["audit", "log"]):
                themes["audit_logs"].append(event.customer_prompt[:200] + "...")
                investigation_types["Audit Review"] += 1
            elif any(word in prompt for word in ["suspicious", "anomal", "unusual"]):
                themes["suspicious_activity"].append(event.customer_prompt[:200] + "...")
                investigation_types["Threat Detection"] += 1
            else:
                themes["other"].append(event.customer_prompt[:200] + "...")
                investigation_types["General Investigation"] += 1
            
            # Count security tools
            if "defender" in prompt or "365 defender" in prompt:
                tools_mentioned["Microsoft 365 Defender"] += 1
            if "sentinel" in prompt:
                tools_mentioned["Microsoft Sentinel"] += 1
            if "entra" in prompt or "azure ad" in prompt:
                tools_mentioned["Microsoft Entra ID"] += 1
        
        # Limit examples per theme
        for theme in themes:
            themes[theme] = themes[theme][:5]  # Top 5 examples
        
        # Store full prompts for each category
        full_prompts = {
            "signin_investigation": [],
            "group_membership": [],
            "audit_logs": [],
            "suspicious_activity": [],
            "general_investigation": []
        }
        
        # Re-iterate to get full prompts with metadata
        for event in events:
            if event.customer_prompt is None:
                continue
                
            prompt = event.customer_prompt.lower()
            prompt_data = {
                'id': event.evaluation_id,
                'prompt': event.customer_prompt,
                'created': event.created_at.strftime("%Y-%m-%d %H:%M") if event.created_at is not None else "Unknown"
            }
            
            if any(word in prompt for word in ["signin", "sign-in", "login"]):
                full_prompts["signin_investigation"].append(prompt_data)
            elif any(word in prompt for word in ["group", "membership", "role"]):
                full_prompts["group_membership"].append(prompt_data)
            elif any(word in prompt for word in ["audit", "log"]):
                full_prompts["audit_logs"].append(prompt_data)
            elif any(word in prompt for word in ["suspicious", "anomal", "unusual"]):
                full_prompts["suspicious_activity"].append(prompt_data)
            else:
                full_prompts["general_investigation"].append(prompt_data)

        return {
            "total_requests": len(events),
            "investigation_types": dict(investigation_types.most_common()),
            "security_tools": dict(tools_mentioned.most_common()),
            "themes": themes,
            "full_prompts": full_prompts,
            "top_patterns": [
                {"pattern": "Signin anomaly detection", "count": investigation_types.get("Signin Analysis", 0)},
                {"pattern": "Access management review", "count": investigation_types.get("Access Management", 0)},
                {"pattern": "Threat investigation", "count": investigation_types.get("Threat Detection", 0)},
                {"pattern": "Audit log analysis", "count": investigation_types.get("Audit Review", 0)}
            ]
        }
    finally:
        db.close()

def get_cluster_analytics() -> List[Dict[str, Any]]:
    """Get cluster analytics for leadership review"""
    db = SessionLocal()
    try:
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).order_by(Cluster.size.desc()).all()
        
        result = []
        for cluster in clusters:
            # Get summary
            summary = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == cluster.id).first()
            
            # Get ALL events in this cluster
            all_events = db.query(TelemetryEvent)\
                           .join(Embedding, TelemetryEvent.id == Embedding.event_id)\
                           .join(ClusterAssignment, Embedding.id == ClusterAssignment.embedding_id)\
                           .filter(ClusterAssignment.cluster_id == cluster.id)\
                           .all()
            
            # Determine cluster theme
            theme = "General Investigation"
            if all_events and all_events[0].customer_prompt is not None:
                prompt_text = all_events[0].customer_prompt.lower()
                if "signin" in prompt_text or "login" in prompt_text:
                    theme = "Signin Investigation"
                elif "group" in prompt_text or "membership" in prompt_text:
                    theme = "Access Management"
                elif "defender" in prompt_text:
                    theme = "Microsoft 365 Defender"
                elif "sentinel" in prompt_text:
                    theme = "Microsoft Sentinel"
                elif "audit" in prompt_text:
                    theme = "Audit Analysis"
            
            cluster_data = {
                "cluster_id": cluster.id,
                "theme": theme,
                "size": cluster.size,
                "summary": summary.summary_text if summary and summary.summary_text is not None else "Analysis in progress...",
                "sample_prompts": [event.customer_prompt[:150] + "..." for event in all_events[:3]],
                "all_prompts": [{"id": event.evaluation_id, "prompt": event.customer_prompt, "created": event.created_at.strftime("%Y-%m-%d %H:%M") if event.created_at is not None else "Unknown"} for event in all_events],
                "priority": "High" if cluster.size > 10 else "Medium" if cluster.size > 5 else "Low",
                "team": "Security Operations" if "security" in theme.lower() else "Analysis Required"
            }
            result.append(cluster_data)
        
        return result
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def main_analytics():
    """Main analytics page - Customer prompt insights"""
    analytics = get_prompt_analytics()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Investigation Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f8f9fa;
                color: #333;
            }}
            .header {{
                background: #2d3748;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .nav {{
                display: flex;
                gap: 20px;
                justify-content: center;
                margin-bottom: 30px;
            }}
            .nav a {{
                background: #4299e1;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 500;
            }}
            .nav a.active {{
                background: #2b6cb0;
            }}
            .nav a:not(.active) {{
                background: #f8f9ff;
                color: #4299e1;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .card h3 {{
                margin-top: 0;
                color: #2d3748;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 10px;
            }}
            .metric {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid #f0f0f0;
            }}
            .metric:last-child {{
                border-bottom: none;
            }}
            .metric-value {{
                font-weight: bold;
                color: #2b6cb0;
            }}
            .pattern-list {{
                list-style: none;
                padding: 0;
            }}
            .pattern-list li {{
                background: #f7fafc;
                margin: 8px 0;
                padding: 12px;
                border-radius: 6px;
                border-left: 4px solid #4299e1;
            }}
            .example {{
                font-size: 0.9em;
                color: #666;
                margin: 5px 0;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
                font-style: italic;
            }}
            .category-section {{
                margin: 20px 0;
                cursor: pointer;
                transition: transform 0.2s ease;
            }}
            .category-section:hover {{
                transform: translateY(-1px);
            }}
            .category-section .expand-indicator {{
                font-size: 1em;
                color: #4299e1;
                transition: transform 0.3s ease;
            }}
            .category-section.expanded .expand-indicator {{
                transform: rotate(180deg);
            }}
            .category-section .all-prompts {{
                display: none;
                margin-top: 15px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }}
            .category-section.expanded .all-prompts {{
                display: block;
            }}
            .prompts-container {{
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: white;
            }}
            .prompt-item {{
                padding: 12px;
                border-bottom: 1px solid #f0f0f0;
                transition: background 0.2s ease;
            }}
            .prompt-item:hover {{
                background: #f7fafc;
            }}
            .prompt-item:last-child {{
                border-bottom: none;
            }}
            .prompt-id {{
                font-size: 0.8em;
                color: #4299e1;
                font-weight: 600;
                margin-bottom: 5px;
            }}
            .prompt-text {{
                color: #2d3748;
                line-height: 1.4;
                margin-bottom: 5px;
            }}
            .prompt-time {{
                font-size: 0.75em;
                color: #718096;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Security Investigation Analytics Dashboard</h1>
            <p>Customer request patterns and investigation insights</p>
        </div>
        
        <div class="nav">
            <a href="/" class="active">📊 Main Analytics</a>
            <a href="/clusters">🎯 Cluster Analysis</a>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📈 Overall Metrics</h3>
                <div class="metric">
                    <span>Total Investigation Requests</span>
                    <span class="metric-value">{analytics['total_requests']}</span>
                </div>
                <div class="metric">
                    <span>Investigation Categories</span>
                    <span class="metric-value">{len(analytics['investigation_types'])}</span>
                </div>
                <div class="metric">
                    <span>Security Tools Used</span>
                    <span class="metric-value">{len(analytics['security_tools'])}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🔍 Investigation Types</h3>
                {chr(10).join([f'<div class="metric"><span>{inv_type}</span><span class="metric-value">{count}</span></div>' for inv_type, count in analytics['investigation_types'].items()])}
            </div>
            
            <div class="card">
                <h3>🛠️ Security Tools</h3>
                {chr(10).join([f'<div class="metric"><span>{tool}</span><span class="metric-value">{count}</span></div>' for tool, count in analytics['security_tools'].items()])}
            </div>
            
            <div class="card">
                <h3>📋 Top Investigation Patterns</h3>
                <ul class="pattern-list">
                    {chr(10).join([f'<li><strong>{pattern["pattern"]}</strong> <span class="metric-value">({pattern["count"]} requests)</span></li>' for pattern in analytics['top_patterns']])}
                </ul>
            </div>
        </div>
        
        <div class="card">
            <h3>💡 Sample Investigation Requests by Category</h3>
            
            <div class="category-section" onclick="toggleCategory('signin')">
                <h4 style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;">
                    🔐 Signin Investigation ({len(analytics['full_prompts']['signin_investigation'])} examples)
                    <span class="expand-indicator">▼</span>
                </h4>
                {chr(10).join([f'<div class="example">{example}</div>' for example in analytics['themes']['signin_investigation'][:3]])}
                <div class="all-prompts" id="signin">
                    <div class="prompts-container">
                        {chr(10).join([f'''<div class="prompt-item">
                            <div class="prompt-id">{prompt['id']}</div>
                            <div class="prompt-text">{prompt['prompt']}</div>
                            <div class="prompt-time">{prompt['created']}</div>
                        </div>''' for prompt in analytics['full_prompts']['signin_investigation']])}
                    </div>
                </div>
            </div>
            
            <div class="category-section" onclick="toggleCategory('group')">
                <h4 style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;">
                    👥 Group Membership ({len(analytics['full_prompts']['group_membership'])} examples)
                    <span class="expand-indicator">▼</span>
                </h4>
                {chr(10).join([f'<div class="example">{example}</div>' for example in analytics['themes']['group_membership'][:3]])}
                <div class="all-prompts" id="group">
                    <div class="prompts-container">
                        {chr(10).join([f'''<div class="prompt-item">
                            <div class="prompt-id">{prompt['id']}</div>
                            <div class="prompt-text">{prompt['prompt']}</div>
                            <div class="prompt-time">{prompt['created']}</div>
                        </div>''' for prompt in analytics['full_prompts']['group_membership']])}
                    </div>
                </div>
            </div>
            
            <div class="category-section" onclick="toggleCategory('suspicious')">
                <h4 style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;">
                    ⚠️ Suspicious Activity ({len(analytics['full_prompts']['suspicious_activity'])} examples)
                    <span class="expand-indicator">▼</span>
                </h4>
                {chr(10).join([f'<div class="example">{example}</div>' for example in analytics['themes']['suspicious_activity'][:3]])}
                <div class="all-prompts" id="suspicious">
                    <div class="prompts-container">
                        {chr(10).join([f'''<div class="prompt-item">
                            <div class="prompt-id">{prompt['id']}</div>
                            <div class="prompt-text">{prompt['prompt']}</div>
                            <div class="prompt-time">{prompt['created']}</div>
                        </div>''' for prompt in analytics['full_prompts']['suspicious_activity']])}
                    </div>
                </div>
            </div>
            
            <div class="category-section" onclick="toggleCategory('audit')">
                <h4 style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;">
                    📝 Audit Logs ({len(analytics['full_prompts']['audit_logs'])} examples)
                    <span class="expand-indicator">▼</span>
                </h4>
                {chr(10).join([f'<div class="example">{example}</div>' for example in analytics['themes']['audit_logs'][:3]])}
                <div class="all-prompts" id="audit">
                    <div class="prompts-container">
                        {chr(10).join([f'''<div class="prompt-item">
                            <div class="prompt-id">{prompt['id']}</div>
                            <div class="prompt-text">{prompt['prompt']}</div>
                            <div class="prompt-time">{prompt['created']}</div>
                        </div>''' for prompt in analytics['full_prompts']['audit_logs']])}
                    </div>
                </div>
            </div>
        </div>        
                 <script>
         function toggleCluster(clusterId) {{
             var cluster = document.getElementById(clusterId).parentElement;
             cluster.classList.toggle('expanded');
         }}
         
         function toggleCategory(categoryId) {{
             var category = document.getElementById(categoryId).parentElement;
             category.classList.toggle('expanded');
         }}
         </script>
    </body>
    </html>
    """
    return html

@app.get("/clusters", response_class=HTMLResponse)
async def cluster_analytics():
    """Cluster analytics page - For leadership review"""
    clusters = get_cluster_analytics()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cluster Analysis - Leadership Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f8f9fa;
                color: #333;
            }}
            .header {{
                background: #2d3748;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .nav {{
                display: flex;
                gap: 20px;
                justify-content: center;
                margin-bottom: 30px;
            }}
            .nav a {{
                background: #4299e1;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 500;
            }}
            .nav a.active {{
                background: #2b6cb0;
            }}
                         .cluster {{
                 background: white;
                 margin: 20px 0;
                 padding: 25px;
                 border-radius: 10px;
                 box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                 border-left: 5px solid #4299e1;
                 cursor: pointer;
                 transition: transform 0.2s ease;
             }}
             .cluster:hover {{
                 transform: translateY(-2px);
                 box-shadow: 0 4px 15px rgba(0,0,0,0.15);
             }}
             .cluster.high {{
                 border-left-color: #e53e3e;
             }}
             .cluster.medium {{
                 border-left-color: #dd6b20;
             }}
             .cluster.low {{
                 border-left-color: #38a169;
             }}
             .cluster-header {{
                 display: flex;
                 justify-content: space-between;
                 align-items: center;
                 margin-bottom: 15px;
             }}
             .cluster-title {{
                 font-size: 1.3em;
                 font-weight: bold;
                 color: #2d3748;
             }}
             .expand-indicator {{
                 font-size: 1.2em;
                 color: #4299e1;
                 transition: transform 0.3s ease;
             }}
             .cluster.expanded .expand-indicator {{
                 transform: rotate(180deg);
             }}
             .priority {{
                 padding: 4px 12px;
                 border-radius: 20px;
                 font-size: 0.8em;
                 font-weight: bold;
                 text-transform: uppercase;
             }}
             .priority.high {{
                 background: #fed7d7;
                 color: #c53030;
             }}
             .priority.medium {{
                 background: #feebc8;
                 color: #c05621;
             }}
             .priority.low {{
                 background: #c6f6d5;
                 color: #2f855a;
             }}
             .cluster-meta {{
                 display: grid;
                 grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 15px;
                 margin-bottom: 20px;
                 padding: 15px;
                 background: #f7fafc;
                 border-radius: 8px;
             }}
             .meta-item {{
                 display: flex;
                 justify-content: space-between;
             }}
             .meta-label {{
                 font-weight: 500;
                 color: #4a5568;
             }}
             .meta-value {{
                 font-weight: bold;
                 color: #2d3748;
             }}
             .summary {{
                 background: #edf2f7;
                 padding: 15px;
                 border-radius: 8px;
                 margin-bottom: 15px;
                 border-left: 3px solid #4299e1;
             }}
             .examples {{
                 margin-top: 15px;
             }}
             .example {{
                 background: #f8f9fa;
                 padding: 10px;
                 margin: 8px 0;
                 border-radius: 6px;
                 font-size: 0.9em;
                 color: #555;
                 border-left: 3px solid #cbd5e0;
             }}
             .all-prompts {{
                 display: none;
                 margin-top: 20px;
                 padding: 15px;
                 background: #f8f9fa;
                 border-radius: 8px;
                 border: 1px solid #e2e8f0;
             }}
             .cluster.expanded .all-prompts {{
                 display: block;
             }}
             .prompts-container {{
                 max-height: 400px;
                 overflow-y: auto;
                 border: 1px solid #e2e8f0;
                 border-radius: 6px;
                 background: white;
             }}
             .prompt-item {{
                 padding: 12px;
                 border-bottom: 1px solid #f0f0f0;
                 transition: background 0.2s ease;
             }}
             .prompt-item:hover {{
                 background: #f7fafc;
             }}
             .prompt-item:last-child {{
                 border-bottom: none;
             }}
             .prompt-id {{
                 font-size: 0.8em;
                 color: #4299e1;
                 font-weight: 600;
                 margin-bottom: 5px;
             }}
             .prompt-text {{
                 color: #2d3748;
                 line-height: 1.4;
                 margin-bottom: 5px;
             }}
             .prompt-time {{
                 font-size: 0.75em;
                 color: #718096;
             }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Investigation Cluster Analysis</h1>
            <p>Leadership dashboard for investigation pattern analysis</p>
        </div>
        
        <div class="nav">
            <a href="/">📊 Main Analytics</a>
            <a href="/clusters" class="active">🎯 Cluster Analysis</a>
        </div>
        
        <div class="summary">
            <h3>📋 Executive Summary</h3>
            <p><strong>{len(clusters)} investigation clusters</strong> identified from customer security requests. Clusters represent common investigation patterns and can help prioritize tooling improvements and training needs.</p>
        </div>
        
        {chr(10).join([f'''
        <div class="cluster {cluster['priority'].lower()}" onclick="toggleCluster('cluster-{cluster['cluster_id']}')">
            <div class="cluster-header">
                <div class="cluster-title">
                    {cluster['theme']} (Cluster {cluster['cluster_id']})
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div class="priority {cluster['priority'].lower()}">{cluster['priority']} Priority</div>
                    <div class="expand-indicator">▼</div>
                </div>
            </div>
            
            <div class="cluster-meta">
                <div class="meta-item">
                    <span class="meta-label">Investigation Count:</span>
                    <span class="meta-value">{cluster['size']} requests</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Responsible Team:</span>
                    <span class="meta-value">{cluster['team']}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Category:</span>
                    <span class="meta-value">{cluster['theme']}</span>
                </div>
            </div>
            
            <div class="summary">
                <strong>Analysis:</strong> {cluster['summary']}
            </div>
            
            <div class="examples">
                <h4>Sample Investigation Requests:</h4>
                {chr(10).join([f'<div class="example">"{prompt}"</div>' for prompt in cluster['sample_prompts']])}
            </div>
            
            <div class="all-prompts" id="cluster-{cluster['cluster_id']}">
                <h4 style="margin: 0 0 15px 0; color: #2d3748;">All Customer Prompts ({cluster['size']} total):</h4>
                <div class="prompts-container">
                    {chr(10).join([f'''<div class="prompt-item">
                        <div class="prompt-id">{prompt['id']}</div>
                        <div class="prompt-text">{prompt['prompt']}</div>
                        <div class="prompt-time">{prompt['created']}</div>
                    </div>''' for prompt in cluster['all_prompts']])}
                </div>
            </div>
        </div>
        ''' for cluster in clusters])}
        
        <script>
        function toggleCluster(clusterId) {{
            var cluster = document.getElementById(clusterId).parentElement;
            cluster.classList.toggle('expanded');
        }}
        </script>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 