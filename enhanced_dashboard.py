#!/usr/bin/env python3
"""
Enhanced Business-Focused Dashboard
- Dynamic clustering with ML + LLM validation
- Real-time analysis and insights
- Business-oriented categorization
"""

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import json
import logging
from typing import Dict, List, Any, Optional
from database import SessionLocal
from models import TelemetryEvent, Embedding, Cluster, ClusterSummary, ClusterAssignment, FailureFingerprint
from clustering import cluster_manager
from summarization import cluster_summarizer
from embeddings import embedding_generator
import uvicorn
from collections import Counter, defaultdict
import re
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Copilot Failure Analysis Dashboard")

class BusinessInsightEngine:
    """Generate business-focused insights from failure data"""
    
    def __init__(self):
        self.business_categories = {
            "authentication_failures": {
                "keywords": ["signin", "login", "auth", "token", "forbidden", "unauthorized"],
                "business_impact": "High - Blocks user productivity",
                "priority": 1
            },
            "integration_failures": {
                "keywords": ["api", "endpoint", "plugin", "connector", "integration"],
                "business_impact": "Medium - Affects specific workflows",
                "priority": 2
            },
            "data_access_failures": {
                "keywords": ["graph", "sharepoint", "teams", "outlook", "office"],
                "business_impact": "High - Impacts daily operations",
                "priority": 1
            },
            "permissions_failures": {
                "keywords": ["permission", "access", "scope", "consent", "admin"],
                "business_impact": "Medium - Security/compliance concern",
                "priority": 2
            },
            "service_availability": {
                "keywords": ["timeout", "unavailable", "server", "503", "502", "500"],
                "business_impact": "Critical - Service disruption",
                "priority": 0
            }
        }
    
    def categorize_failure(self, event: TelemetryEvent) -> Dict[str, Any]:
        """Categorize a failure for business impact"""
        text = f"{event.customer_prompt or ''} {json.dumps(event.skill_output) if event.skill_output else ''}".lower()
        
        for category, config in self.business_categories.items():
            if any(keyword in text for keyword in config["keywords"]):
                return {
                    "category": category,
                    "business_impact": config["business_impact"],
                    "priority": config["priority"]
                }
        
        return {
            "category": "general_failure",
            "business_impact": "Low - Needs investigation",
            "priority": 3
        }
    
    def generate_business_insights(self) -> Dict[str, Any]:
        """Generate comprehensive business insights"""
        db = SessionLocal()
        try:
            events = db.query(TelemetryEvent).all()
            
            # Categorize all failures
            categories = defaultdict(list)
            priority_breakdown = defaultdict(int)
            recent_failures = defaultdict(int)
            
            # Get recent failures (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            for event in events:
                category_info = self.categorize_failure(event)
                categories[category_info["category"]].append({
                    "event": event,
                    "business_impact": category_info["business_impact"],
                    "priority": category_info["priority"]
                })
                priority_breakdown[category_info["priority"]] += 1
                
                if event.created_at and event.created_at > week_ago:
                    recent_failures[category_info["category"]] += 1
            
            # Generate insights
            total_failures = len(events)
            critical_failures = priority_breakdown[0]
            high_priority = priority_breakdown[1]
            
            return {
                "total_failures": total_failures,
                "critical_failures": critical_failures,
                "high_priority_failures": high_priority,
                "categories": dict(categories),
                "priority_breakdown": dict(priority_breakdown),
                "recent_trends": dict(recent_failures),
                "business_recommendations": self._generate_recommendations(categories, priority_breakdown)
            }
        finally:
            db.close()
    
    def _generate_recommendations(self, categories: dict, priority_breakdown: dict) -> List[Dict[str, str]]:
        """Generate business recommendations"""
        recommendations = []
        
        if priority_breakdown[0] > 0:  # Critical failures
            recommendations.append({
                "type": "urgent",
                "title": "Critical Service Issues Detected",
                "description": f"{priority_breakdown[0]} critical failures affecting service availability",
                "action": "Immediate escalation to infrastructure team required"
            })
        
        if len(categories.get("authentication_failures", [])) > 10:
            recommendations.append({
                "type": "security",
                "title": "Authentication Pattern Detected",
                "description": "High volume of authentication failures may indicate security issues",
                "action": "Review authentication mechanisms and user access patterns"
            })
        
        if len(categories.get("integration_failures", [])) > 20:
            recommendations.append({
                "type": "integration",
                "title": "Integration Stability Issues",
                "description": "Multiple API/plugin failures suggest integration problems",
                "action": "Audit third-party integrations and API reliability"
            })
        
        return recommendations

insight_engine = BusinessInsightEngine()

def get_enhanced_prompt_analytics() -> Dict[str, Any]:
    """Enhanced prompt analytics with business context"""
    db = SessionLocal()
    try:
        events = db.query(TelemetryEvent).all()
        
        # Business-focused categorization
        business_insights = insight_engine.generate_business_insights()
        
        # User intent analysis
        user_intents = {
            "productivity_tasks": [],
            "data_analysis": [],
            "communication": [],
            "file_management": [],
            "calendar_scheduling": [],
            "security_investigation": []
        }
        
        for event in events:
            if not event.customer_prompt:
                continue
                
            prompt = event.customer_prompt.lower()
            
            # Categorize by user intent
            if any(word in prompt for word in ["meeting", "calendar", "schedule", "appointment"]):
                user_intents["calendar_scheduling"].append(event.customer_prompt[:200])
            elif any(word in prompt for word in ["file", "document", "upload", "download", "share"]):
                user_intents["file_management"].append(event.customer_prompt[:200])
            elif any(word in prompt for word in ["email", "message", "teams", "chat"]):
                user_intents["communication"].append(event.customer_prompt[:200])
            elif any(word in prompt for word in ["analyze", "report", "data", "chart", "graph"]):
                user_intents["data_analysis"].append(event.customer_prompt[:200])
            elif any(word in prompt for word in ["security", "audit", "signin", "suspicious"]):
                user_intents["security_investigation"].append(event.customer_prompt[:200])
            else:
                user_intents["productivity_tasks"].append(event.customer_prompt[:200])
        
        return {
            **business_insights,
            "user_intents": user_intents,
            "intent_distribution": {k: len(v) for k, v in user_intents.items()}
        }
    finally:
        db.close()

def get_ml_cluster_analytics() -> Dict[str, Any]:
    """ML-powered cluster analytics with LLM validation"""
    db = SessionLocal()
    try:
        # Get clustering metrics
        embeddings = db.query(Embedding).all()
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
        
        if not embeddings or not clusters:
            return {"error": "No clustering data available"}
        
        # Calculate clustering quality metrics
        embedding_vectors = np.array([emb.embedding_vector for emb in embeddings])
        cluster_labels = []
        
        for emb in embeddings:
            assignment = db.query(ClusterAssignment).filter(
                ClusterAssignment.embedding_id == emb.id
            ).first()
            cluster_labels.append(assignment.cluster_id if assignment else -1)
        
        silhouette_avg = silhouette_score(embedding_vectors, cluster_labels)
        
        # Analyze clusters with business context
        cluster_analysis = []
        for cluster in clusters:
            # Get events in cluster
            events = db.query(TelemetryEvent).join(
                Embedding, TelemetryEvent.id == Embedding.event_id
            ).join(
                ClusterAssignment, Embedding.id == ClusterAssignment.embedding_id
            ).filter(ClusterAssignment.cluster_id == cluster.id).all()
            
            # Business categorization
            business_categories = []
            for event in events:
                category_info = insight_engine.categorize_failure(event)
                business_categories.append(category_info["category"])
            
            business_category_counts = Counter(business_categories)
            
            # Get LLM summary
            summary = db.query(ClusterSummary).filter(
                ClusterSummary.cluster_id == cluster.id
            ).first()
            
            cluster_analysis.append({
                "cluster_id": cluster.id,
                "size": cluster.size,
                "business_categories": dict(business_category_counts),
                "primary_business_impact": insight_engine.business_categories.get(
                    business_category_counts.most_common(1)[0][0] if business_category_counts else "general_failure",
                    {}
                ).get("business_impact", "Unknown"),
                "llm_summary": summary.summary_text if summary else "Not generated",
                "root_cause": summary.root_cause if summary else "Not analyzed",
                "recommendations": summary.recommendations if summary and summary.recommendations else [],
                "sample_prompts": [event.customer_prompt[:100] + "..." for event in events[:3] if event.customer_prompt]
            })
        
        return {
            "clustering_quality": {
                "silhouette_score": silhouette_avg,
                "num_clusters": len(clusters),
                "total_events": len(embeddings),
                "noise_points": len([c for c in clusters if c.is_noise])
            },
            "clusters": cluster_analysis,
            "business_summary": {
                "critical_clusters": len([c for c in cluster_analysis if "Critical" in c.get("primary_business_impact", "")]),
                "authentication_issues": len([c for c in cluster_analysis if "authentication_failures" in c["business_categories"]]),
                "integration_issues": len([c for c in cluster_analysis if "integration_failures" in c["business_categories"]])
            }
        }
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Main dashboard: LLM-driven cluster summary, expandable, no static intent analysis"""
    db = SessionLocal()
    try:
        analytics = get_ml_cluster_analytics()
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
        cluster_summaries = {s.cluster_id: s for s in db.query(ClusterSummary).all()}
        cluster_prompts = {}
        for cluster in clusters:
            event_prompts = db.query(TelemetryEvent.customer_prompt).join(
                Embedding, TelemetryEvent.id == Embedding.event_id
            ).join(
                ClusterAssignment, Embedding.id == ClusterAssignment.embedding_id
            ).filter(ClusterAssignment.cluster_id == cluster.id).all()
            cluster_prompts[cluster.id] = [p[0] for p in event_prompts if p[0]]
    finally:
        db.close()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Copilot - Customer Insights</title>
        <meta charset='utf-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1'>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; min-height: 100vh; }}
            .header {{ background: rgba(255,255,255,0.95); color: #2d3748; padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }}
            .nav {{ display: flex; gap: 20px; justify-content: center; margin-bottom: 30px; }}
            .nav a {{ background: rgba(255,255,255,0.9); color: #4299e1; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            .nav a:hover, .nav a.active {{ background: #4299e1; color: white; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(66,153,225,0.3); }}
            .cluster-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
            .cluster-card {{ background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); transition: transform 0.3s ease; position: relative; }}
            .cluster-card:hover {{ transform: translateY(-5px); }}
            .cluster-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 2px solid #e2e8f0; }}
            .cluster-size {{ background: #10b981; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
            .customer-intent {{ font-size: 1.3em; font-weight: bold; color: #065f46; margin-bottom: 8px; }}
            .intent-description {{ color: #374151; font-size: 1.1em; line-height: 1.5; margin-bottom: 15px; }}
            .expand-btn {{ background: #10b981; color: white; border: none; border-radius: 6px; padding: 6px 16px; cursor: pointer; font-size: 1em; margin-bottom: 10px; }}
            .expand-btn:hover {{ background: #059669; }}
            .prompts-list {{ display: none; max-height: 300px; overflow-y: auto; background: #f0fdf4; border-radius: 8px; margin-top: 10px; padding: 10px; }}
            .prompts-list.expanded {{ display: block; }}
            .prompt-item {{ padding: 6px 0; border-bottom: 1px solid #d1fae5; font-size: 0.98em; }}
            .prompt-item:last-child {{ border-bottom: none; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Customer Insights</h1>
            <p>Understanding what customers are asking Security Copilot about, based on AI analysis of prompts.</p>
        </div>
        <div class="nav">
            <a href="/" class="active">� Customer Insights</a>
            <a href="/clusters">⚠️ Error Analysis</a>
        </div>
        <div class="cluster-grid">
            {chr(10).join([
                f'''<div class="cluster-card">
                    <div class="cluster-header">
                        <span class="customer-intent">{(cluster_summaries.get(cluster['cluster_id']).label or 'Customer Intent') if cluster_summaries.get(cluster['cluster_id']) else 'Customer Intent'}</span>
                        <div class="cluster-size">{cluster['size']} customers</div>
                    </div>
                    <div class="intent-description">
                        {cluster['llm_summary'][:300]}{'...' if len(cluster['llm_summary']) > 300 else ''}
                    </div>
                    <button class="expand-btn" onclick="togglePrompts('prompts-{cluster['cluster_id']}')">View Sample Questions</button>
                    <div class="prompts-list" id="prompts-{cluster['cluster_id']}">
                        {chr(10).join([f'<div class="prompt-item">"{p[:150]}..."</div>' for p in cluster_prompts.get(cluster['cluster_id'], [])[:8]])}
                    </div>
                </div>''' for cluster in analytics['clusters']])}
        </div>
        <script>
            function togglePrompts(id) {{
                var el = document.getElementById(id);
                if (el.classList.contains('expanded')) {{
                    el.classList.remove('expanded');
                }} else {{
                    el.classList.add('expanded');
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html

@app.get("/clusters", response_class=HTMLResponse)
async def ml_cluster_dashboard():
    """ML Cluster Analysis Dashboard with expandable clusters and LLM labels"""
    db = SessionLocal()
    try:
        analytics = get_ml_cluster_analytics()
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
        cluster_summaries = {s.cluster_id: s for s in db.query(ClusterSummary).all()}
        cluster_prompts = {}
        for cluster in clusters:
            # Get all prompts for this cluster
            event_prompts = db.query(TelemetryEvent.customer_prompt).join(
                Embedding, TelemetryEvent.id == Embedding.event_id
            ).join(
                ClusterAssignment, Embedding.id == ClusterAssignment.embedding_id
            ).filter(ClusterAssignment.cluster_id == cluster.id).all()
            cluster_prompts[cluster.id] = [p[0] for p in event_prompts if p[0]]
    finally:
        db.close()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Copilot - Error Analysis</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: #333; min-height: 100vh; }}
            .header {{ background: rgba(255, 255, 255, 0.95); color: #2d3748; padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }}
            .nav {{ display: flex; gap: 20px; justify-content: center; margin-bottom: 30px; }}
            .nav a {{ background: rgba(255, 255, 255, 0.9); color: #dc2626; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            .nav a:hover, .nav a.active {{ background: #dc2626; color: white; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(220,38,38,0.3); }}
            .cluster-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }}
            .cluster-card {{ background: rgba(255, 255, 255, 0.95); padding: 25px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); transition: transform 0.3s ease; position: relative; }}
            .cluster-card:hover {{ transform: translateY(-5px); }}
            .cluster-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 2px solid #fecaca; }}
            .cluster-size {{ background: #dc2626; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
            .error-type {{ font-size: 1.3em; font-weight: bold; color: #991b1b; margin-bottom: 8px; }}
            .root-cause {{ background: #fef2f2; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #dc2626; }}
            .recommendations {{ background: #f0fdf4; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #10b981; }}
            .expand-btn {{ background: #dc2626; color: white; border: none; border-radius: 6px; padding: 6px 16px; cursor: pointer; font-size: 1em; margin-bottom: 10px; }}
            .expand-btn:hover {{ background: #991b1b; }}
            .prompts-list {{ display: none; max-height: 300px; overflow-y: auto; background: #fef2f2; border-radius: 8px; margin-top: 10px; padding: 10px; }}
            .prompts-list.expanded {{ display: block; }}
            .prompt-item {{ padding: 6px 0; border-bottom: 1px solid #fecaca; font-size: 0.98em; }}
            .prompt-item:last-child {{ border-bottom: none; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚠️ Error Analysis</h1>
            <p>AI-powered root cause analysis and recommended fixes for Security Copilot failures.</p>
        </div>
        <div class="nav">
            <a href="/">🔍 Customer Insights</a>
            <a href="/clusters" class="active">⚠️ Error Analysis</a>
            <a href="/api/recluster" onclick="recluster()">🔄 Re-run Analysis</a>
        </div>
            <a href="/">📊 Business Insights</a>
            <a href="/clusters" class="active">🎯 ML Cluster Analysis</a>
            <a href="/api/recluster" onclick="recluster()">🔄 Re-run Clustering</a>
        </div>
        <div class="cluster-grid">
            {chr(10).join([
                f'''<div class="cluster-card">
                    <div class="cluster-header">
                        <span class="error-type">{(cluster_summaries.get(cluster['cluster_id']).label or 'Error Pattern') if cluster_summaries.get(cluster['cluster_id']) else 'Error Pattern'}</span>
                        <div class="cluster-size">{cluster['size']} failures</div>
                    </div>
                    <div class="root-cause">
                        <strong>🔍 Root Cause:</strong><br>
                        {cluster['root_cause'][:300]}{'...' if len(cluster['root_cause']) > 300 else ''}
                    </div>
                    <div class="recommendations">
                        <strong>🛠️ Recommended Fixes:</strong><br>
                        {'<br>• '.join([''] + cluster.get('recommendations', ['No recommendations available']))}
                    </div>
                    <button class="expand-btn" onclick="togglePrompts('prompts-{cluster['cluster_id']}')">View Failed Prompts</button>
                    <div class="prompts-list" id="prompts-{cluster['cluster_id']}">
                        {chr(10).join([f'<div class="prompt-item">❌ "{p[:150]}..."</div>' for p in cluster_prompts.get(cluster['cluster_id'], [])[:8]])}
                    </div>
                </div>''' for cluster in analytics['clusters']])}
        </div>
        <script>
            function togglePrompts(id) {{
                var el = document.getElementById(id);
                if (el.classList.contains('expanded')) {{
                    el.classList.remove('expanded');
                }} else {{
                    el.classList.add('expanded');
                }}
            }}
            function recluster() {{
                if (confirm('Re-run clustering analysis? This may take a few minutes.')) {{
                    fetch('/api/recluster', {{method: 'POST'}})
                        .then(response => response.json())
                        .then(data => {{
                            alert('Clustering completed: ' + data.message);
                            location.reload();
                        }});
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html

@app.post("/api/refresh")
async def refresh_analysis(background_tasks: BackgroundTasks):
    """Refresh the entire analysis pipeline"""
    background_tasks.add_task(run_full_analysis)
    return {"message": "Analysis refresh initiated"}

@app.post("/api/recluster")
async def recluster_data():
    """Re-run clustering with current embeddings"""
    try:
        # Run clustering
        result = cluster_manager.perform_clustering()
        
        # Generate summaries for new clusters
        db = SessionLocal()
        try:
            clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
            for cluster in clusters:
                existing_summary = db.query(ClusterSummary).filter(
                    ClusterSummary.cluster_id == cluster.id
                ).first()
                if not existing_summary:
                    cluster_summarizer.generate_cluster_summary(cluster.id)
        finally:
            db.close()
        
        return {"message": f"Clustering completed. Found {result.get('num_clusters', 0)} clusters"}
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        return {"error": str(e)}

def run_full_analysis():
    """Run the complete analysis pipeline"""
    try:
        logger.info("Starting full analysis pipeline")
        
        # 1. Generate missing embeddings
        db = SessionLocal()
        event_ids = [id[0] for id in db.query(TelemetryEvent.id).outerjoin(Embedding).filter(Embedding.id.is_(None)).all()]
        db.close()
        
        if event_ids:
            embedding_generator.generate_embeddings_batch(event_ids, use_openai=False)
        
        # 2. Re-run clustering
        cluster_manager.perform_clustering()
        
        # 3. Generate summaries
        db = SessionLocal()
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
        for cluster in clusters:
            existing_summary = db.query(ClusterSummary).filter(
                ClusterSummary.cluster_id == cluster.id
            ).first()
            if not existing_summary:
                cluster_summarizer.generate_cluster_summary(cluster.id)
        db.close()
        
        logger.info("Full analysis pipeline completed")
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
