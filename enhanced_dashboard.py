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
            /* Microsoft Fluent Design System - Enhanced & Dynamic */
            * {{ box-sizing: border-box; }}
            
            body {{ 
                font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif; 
                margin: 0; 
                padding: 0; 
                background: linear-gradient(135deg, #0078d4 0%, #106ebe 100%); 
                color: #323130; 
                min-height: 100vh; 
                font-size: 14px;
                line-height: 1.4;
                overflow-x: hidden;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{ 
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
                color: #323130; 
                padding: 40px; 
                border-radius: 12px; 
                margin-bottom: 32px; 
                text-align: center; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.12); 
                border: 1px solid rgba(255,255,255,0.2);
                position: relative;
                overflow: hidden;
            }}
            
            .header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
                animation: shimmer 3s infinite;
            }}
            
            @keyframes shimmer {{
                0% {{ left: -100%; }}
                100% {{ left: 100%; }}
            }}
            
            .header h1 {{ 
                margin: 0 0 12px 0; 
                font-size: 36px; 
                font-weight: 700; 
                color: #106ebe; 
                position: relative;
                z-index: 1;
            }}
            
            .header p {{ 
                margin: 0; 
                color: #605e5c; 
                font-size: 18px; 
                position: relative;
                z-index: 1;
            }}
            
            .nav {{ 
                display: flex; 
                gap: 8px; 
                justify-content: center; 
                margin-bottom: 32px; 
                background: rgba(255,255,255,0.1);
                padding: 8px;
                border-radius: 50px;
                backdrop-filter: blur(10px);
                width: fit-content;
                margin-left: auto;
                margin-right: auto;
            }}
            
            .nav a {{ 
                background: transparent; 
                color: #ffffff; 
                padding: 14px 28px; 
                text-decoration: none; 
                border-radius: 25px; 
                font-weight: 600; 
                font-size: 14px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
                position: relative;
                overflow: hidden;
            }}
            
            .nav a::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: rgba(255,255,255,0.2);
                transition: left 0.3s ease;
            }}
            
            .nav a:hover::before {{
                left: 0;
            }}
            
            .nav a.active {{ 
                background: #ffffff; 
                color: #0078d4; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            
            .stats-bar {{
                display: flex;
                gap: 24px;
                margin-bottom: 32px;
                justify-content: center;
            }}
            
            .stat-card {{
                background: rgba(255,255,255,0.95);
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
                min-width: 140px;
                transition: transform 0.3s ease;
            }}
            
            .stat-card:hover {{
                transform: translateY(-4px);
            }}
            
            .stat-number {{
                font-size: 32px;
                font-weight: 700;
                color: #0078d4;
                margin-bottom: 8px;
            }}
            
            .stat-label {{
                color: #605e5c;
                font-size: 14px;
                font-weight: 500;
            }}
            
            .cluster-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); 
                gap: 24px; 
            }}
            
            .cluster-card {{ 
                background: #ffffff; 
                padding: 0; 
                border-radius: 16px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
                border: 1px solid rgba(255,255,255,0.2);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); 
                position: relative; 
                overflow: hidden;
            }}
            
            .cluster-card:hover {{ 
                transform: translateY(-8px) scale(1.02); 
                box-shadow: 0 20px 40px rgba(0,0,0,0.15); 
            }}
            
            .cluster-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #0078d4, #106ebe, #005a9e);
            }}
            
            .cluster-content {{
                padding: 28px;
            }}
            
            .cluster-header {{ 
                display: flex; 
                justify-content: space-between; 
                align-items: flex-start; 
                margin-bottom: 20px; 
            }}
            
            .cluster-size {{ 
                background: linear-gradient(135deg, #0078d4, #106ebe); 
                color: #ffffff; 
                padding: 8px 16px; 
                border-radius: 20px; 
                font-weight: 600; 
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                box-shadow: 0 2px 8px rgba(0,120,212,0.3);
            }}
            
            .customer-intent {{ 
                font-size: 20px; 
                font-weight: 600; 
                color: #323130; 
                margin-bottom: 12px; 
                flex: 1;
                margin-right: 20px;
                line-height: 1.3;
            }}
            
            .intent-description {{ 
                color: #605e5c; 
                font-size: 15px; 
                line-height: 1.6; 
                margin-bottom: 20px; 
            }}
            
            .expand-btn {{ 
                background: linear-gradient(135deg, #0078d4, #106ebe); 
                color: #ffffff; 
                border: none; 
                border-radius: 8px; 
                padding: 12px 24px; 
                cursor: pointer; 
                font-size: 14px; 
                font-weight: 600;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                margin-bottom: 16px; 
                position: relative;
                overflow: hidden;
            }}
            
            .expand-btn::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s ease;
            }}
            
            .expand-btn:hover::before {{
                left: 100%;
            }}
            
            .expand-btn:hover {{ 
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,120,212,0.3);
            }}
            
            /* Modal Overlay */
            .modal-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.6);
                backdrop-filter: blur(8px);
                z-index: 1000;
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s ease;
            }}
            
            .modal-overlay.active {{
                opacity: 1;
                visibility: visible;
            }}
            
            /* Modal Content */
            .modal {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%) scale(0.8);
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 700px;
                width: 90%;
                max-height: 80vh;
                overflow: hidden;
                z-index: 1001;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            .modal.active {{
                transform: translate(-50%, -50%) scale(1);
            }}
            
            .modal-header {{
                background: linear-gradient(135deg, #0078d4, #106ebe);
                color: white;
                padding: 24px;
                position: relative;
            }}
            
            .modal-title {{
                font-size: 20px;
                font-weight: 600;
                margin: 0;
            }}
            
            .modal-close {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.2);
                border: none;
                color: white;
                font-size: 24px;
                width: 36px;
                height: 36px;
                border-radius: 50%;
                cursor: pointer;
                transition: background 0.2s ease;
            }}
            
            .modal-close:hover {{
                background: rgba(255,255,255,0.3);
            }}
            
            .modal-body {{
                padding: 24px;
                max-height: 400px;
                overflow-y: auto;
            }}
            
            .prompt-item-modal {{ 
                padding: 16px 0; 
                border-bottom: 1px solid #f3f2f1; 
                font-size: 14px; 
                color: #323130;
                line-height: 1.5;
                transition: background 0.2s ease;
            }}
            
            .prompt-item-modal:hover {{
                background: #f8f9fa;
                padding-left: 12px;
                border-radius: 8px;
            }}
            
            .prompt-item-modal:last-child {{ 
                border-bottom: none; 
            }}
            
            .prompt-number {{
                display: inline-block;
                background: #0078d4;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                text-align: center;
                line-height: 24px;
                font-size: 12px;
                font-weight: 600;
                margin-right: 12px;
            }}
            
            /* Custom scrollbar */
            .modal-body::-webkit-scrollbar {{ 
                width: 8px; 
            }}
            .modal-body::-webkit-scrollbar-track {{ 
                background: #f1f1f1; 
                border-radius: 4px;
            }}
            .modal-body::-webkit-scrollbar-thumb {{ 
                background: #c8c6c4; 
                border-radius: 4px; 
            }}
            .modal-body::-webkit-scrollbar-thumb:hover {{ 
                background: #a19f9d; 
            }}
            
            /* Loading Animation */
            .loading {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #0078d4;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 8px;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            /* Responsive Design */
            @media (max-width: 768px) {{
                .container {{
                    padding: 12px;
                }}
                
                .cluster-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .stats-bar {{
                    flex-wrap: wrap;
                    gap: 16px;
                }}
                
                .modal {{
                    width: 95%;
                    margin: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Customer Insights</h1>
                <p>Understanding what customers are asking Security Copilot about, based on AI analysis of prompts.</p>
            </div>
            <div class="nav">
                <a href="/" class="active">Customer Insights</a>
                <a href="/clusters">Error Analysis</a>
            </div>
            <div class="stats-bar">
                <div class="stat-card">
                    <div class="stat-number">{len(analytics['clusters'])}</div>
                    <div class="stat-label">Intent Categories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{sum(cluster['size'] for cluster in analytics['clusters'])}</div>
                    <div class="stat-label">Total Interactions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{max(cluster['size'] for cluster in analytics['clusters']) if analytics['clusters'] else 0}</div>
                    <div class="stat-label">Largest Category</div>
                </div>
            </div>
            <div class="cluster-grid">
                {chr(10).join([
                    f'''<div class="cluster-card">
                        <div class="cluster-content">
                            <div class="cluster-header">
                                <div class="customer-intent">{(cluster_summaries.get(cluster['cluster_id']).label or 'Customer Intent') if cluster_summaries.get(cluster['cluster_id']) else 'Customer Intent'}</div>
                                <div class="cluster-size">{cluster['size']} customers</div>
                            </div>
                            <div class="intent-description">
                                {cluster['llm_summary'][:250]}{'...' if len(cluster['llm_summary']) > 250 else ''}
                            </div>
                            <button class="expand-btn" onclick="openModal('modal-{cluster['cluster_id']}')">
                                <span class="loading" style="display: none;"></span>
                                View Sample Questions
                            </button>
                        </div>
                    </div>''' for cluster in analytics['clusters']])}
            </div>
        </div>
        
        <!-- Modals for each cluster -->
        {chr(10).join([
            f'''<div class="modal-overlay" id="overlay-{cluster['cluster_id']}">
                <div class="modal" id="modal-{cluster['cluster_id']}">
                    <div class="modal-header">
                        <h3 class="modal-title">{(cluster_summaries.get(cluster['cluster_id']).label or 'Customer Questions') if cluster_summaries.get(cluster['cluster_id']) else 'Customer Questions'}</h3>
                        <button class="modal-close" onclick="closeModal('modal-{cluster['cluster_id']}')">&times;</button>
                    </div>
                    <div class="modal-body">
                        {chr(10).join([f'<div class="prompt-item-modal"><span class="prompt-number">{i+1}</span>"{p}"</div>' for i, p in enumerate(cluster_prompts.get(cluster['cluster_id'], [])[:15])])}
                    </div>
                </div>
            </div>''' for cluster in analytics['clusters']])}
        
        <script>
            function openModal(modalId) {{
                const overlay = document.getElementById('overlay-' + modalId.split('-')[1]);
                const modal = document.getElementById(modalId);
                
                overlay.classList.add('active');
                modal.classList.add('active');
                
                // Prevent body scroll
                document.body.style.overflow = 'hidden';
            }}
            
            function closeModal(modalId) {{
                const overlay = document.getElementById('overlay-' + modalId.split('-')[1]);
                const modal = document.getElementById(modalId);
                
                overlay.classList.remove('active');
                modal.classList.remove('active');
                
                // Restore body scroll
                document.body.style.overflow = 'auto';
            }}
            
            // Close modal when clicking overlay
            document.addEventListener('click', function(e) {{
                if (e.target.classList.contains('modal-overlay')) {{
                    const modalId = e.target.querySelector('.modal').id;
                    closeModal(modalId);
                }}
            }});
            
            // Close modal with Escape key
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') {{
                    const activeModal = document.querySelector('.modal-overlay.active');
                    if (activeModal) {{
                        const modalId = activeModal.querySelector('.modal').id;
                        closeModal(modalId);
                    }}
                }}
            }});
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
            /* Microsoft Fluent Design System - Error Analysis Enhanced */
            * {{ box-sizing: border-box; }}
            
            body {{ 
                font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif; 
                margin: 0; 
                padding: 0; 
                background: linear-gradient(135deg, #d83b01 0%, #c73e1d 100%); 
                color: #323130; 
                min-height: 100vh; 
                font-size: 14px;
                line-height: 1.4;
                overflow-x: hidden;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{ 
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
                color: #323130; 
                padding: 40px; 
                border-radius: 12px; 
                margin-bottom: 32px; 
                text-align: center; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.12); 
                border: 1px solid rgba(255,255,255,0.2);
                position: relative;
                overflow: hidden;
            }}
            
            .header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
                animation: shimmer 3s infinite;
            }}
            
            @keyframes shimmer {{
                0% {{ left: -100%; }}
                100% {{ left: 100%; }}
            }}
            
            .header h1 {{ 
                margin: 0 0 12px 0; 
                font-size: 36px; 
                font-weight: 700; 
                color: #d83b01; 
                position: relative;
                z-index: 1;
            }}
            
            .header p {{ 
                margin: 0; 
                color: #605e5c; 
                font-size: 18px; 
                position: relative;
                z-index: 1;
            }}
            
            .nav {{ 
                display: flex; 
                gap: 8px; 
                justify-content: center; 
                margin-bottom: 32px; 
                background: rgba(255,255,255,0.1);
                padding: 8px;
                border-radius: 50px;
                backdrop-filter: blur(10px);
                width: fit-content;
                margin-left: auto;
                margin-right: auto;
            }}
            
            .nav a {{ 
                background: transparent; 
                color: #ffffff; 
                padding: 14px 28px; 
                text-decoration: none; 
                border-radius: 25px; 
                font-weight: 600; 
                font-size: 14px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
                position: relative;
                overflow: hidden;
            }}
            
            .nav a::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: rgba(255,255,255,0.2);
                transition: left 0.3s ease;
            }}
            
            .nav a:hover::before {{
                left: 0;
            }}
            
            .nav a.active {{ 
                background: #ffffff; 
                color: #d83b01; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            
            .stats-bar {{
                display: flex;
                gap: 24px;
                margin-bottom: 32px;
                justify-content: center;
            }}
            
            .stat-card {{
                background: rgba(255,255,255,0.95);
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
                min-width: 140px;
                transition: transform 0.3s ease;
            }}
            
            .stat-card:hover {{
                transform: translateY(-4px);
            }}
            
            .stat-number {{
                font-size: 32px;
                font-weight: 700;
                color: #d83b01;
                margin-bottom: 8px;
            }}
            
            .stat-label {{
                color: #605e5c;
                font-size: 14px;
                font-weight: 500;
            }}
            
            .cluster-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); 
                gap: 24px; 
            }}
            
            .cluster-card {{ 
                background: #ffffff; 
                padding: 0; 
                border-radius: 16px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
                border: 1px solid rgba(255,255,255,0.2);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); 
                position: relative; 
                overflow: hidden;
            }}
            
            .cluster-card:hover {{ 
                transform: translateY(-8px) scale(1.02); 
                box-shadow: 0 20px 40px rgba(0,0,0,0.15); 
            }}
            
            .cluster-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #d83b01, #c73e1d, #a02d00);
            }}
            
            .cluster-content {{
                padding: 28px;
            }}
            
            .cluster-header {{ 
                display: flex; 
                justify-content: space-between; 
                align-items: flex-start; 
                margin-bottom: 20px; 
            }}
            
            .cluster-size {{ 
                background: linear-gradient(135deg, #d83b01, #c73e1d); 
                color: #ffffff; 
                padding: 8px 16px; 
                border-radius: 20px; 
                font-weight: 600; 
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                box-shadow: 0 2px 8px rgba(216,59,1,0.3);
            }}
            
            .error-type {{ 
                font-size: 20px; 
                font-weight: 600; 
                color: #323130; 
                margin-bottom: 12px; 
                flex: 1;
                margin-right: 20px;
                line-height: 1.3;
            }}
            
            .root-cause {{ 
                background: linear-gradient(135deg, #fef9f9, #fdeaea); 
                padding: 20px; 
                border-radius: 12px; 
                margin-bottom: 20px; 
                border-left: 4px solid #d83b01; 
                color: #323130;
                font-size: 14px;
                position: relative;
            }}
            
            .root-cause::before {{
                content: '';
                position: absolute;
                top: 12px;
                right: 12px;
                width: 24px;
                height: 24px;
                background: #d83b01;
                border-radius: 50%;
                opacity: 0.1;
            }}
            
            .root-cause strong {{ 
                color: #d83b01; 
                font-weight: 600;
                font-size: 15px;
            }}
            
            .recommendations {{ 
                background: linear-gradient(135deg, #f8fff8, #eaf7ea); 
                padding: 20px; 
                border-radius: 12px; 
                margin-bottom: 20px; 
                border-left: 4px solid #107c10; 
                color: #323130;
                font-size: 14px;
                position: relative;
            }}
            
            .recommendations::before {{
                content: '';
                position: absolute;
                top: 12px;
                right: 12px;
                width: 24px;
                height: 24px;
                background: #107c10;
                border-radius: 50%;
                opacity: 0.1;
            }}
            
            .recommendations strong {{ 
                color: #107c10; 
                font-weight: 600;
                font-size: 15px;
            }}
            
            .expand-btn {{ 
                background: linear-gradient(135deg, #d83b01, #c73e1d); 
                color: #ffffff; 
                border: none; 
                border-radius: 8px; 
                padding: 12px 24px; 
                cursor: pointer; 
                font-size: 14px; 
                font-weight: 600;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                margin-bottom: 16px; 
                position: relative;
                overflow: hidden;
            }}
            
            .expand-btn::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s ease;
            }}
            
            .expand-btn:hover::before {{
                left: 100%;
            }}
            
            .expand-btn:hover {{ 
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(216,59,1,0.3);
            }}
            
            /* Modal Overlay */
            .modal-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.6);
                backdrop-filter: blur(8px);
                z-index: 1000;
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s ease;
            }}
            
            .modal-overlay.active {{
                opacity: 1;
                visibility: visible;
            }}
            
            /* Modal Content */
            .modal {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%) scale(0.8);
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 700px;
                width: 90%;
                max-height: 80vh;
                overflow: hidden;
                z-index: 1001;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            .modal.active {{
                transform: translate(-50%, -50%) scale(1);
            }}
            
            .modal-header {{
                background: linear-gradient(135deg, #d83b01, #c73e1d);
                color: white;
                padding: 24px;
                position: relative;
            }}
            
            .modal-title {{
                font-size: 20px;
                font-weight: 600;
                margin: 0;
            }}
            
            .modal-close {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.2);
                border: none;
                color: white;
                font-size: 24px;
                width: 36px;
                height: 36px;
                border-radius: 50%;
                cursor: pointer;
                transition: background 0.2s ease;
            }}
            
            .modal-close:hover {{
                background: rgba(255,255,255,0.3);
            }}
            
            .modal-body {{
                padding: 24px;
                max-height: 400px;
                overflow-y: auto;
            }}
            
            .prompt-item-modal {{ 
                padding: 16px 0; 
                border-bottom: 1px solid #f3f2f1; 
                font-size: 14px; 
                color: #323130;
                line-height: 1.5;
                transition: background 0.2s ease;
            }}
            
            .prompt-item-modal:hover {{
                background: #fef9f9;
                padding-left: 12px;
                border-radius: 8px;
            }}
            
            .prompt-item-modal:last-child {{ 
                border-bottom: none; 
            }}
            
            .prompt-number {{
                display: inline-block;
                background: #d83b01;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                text-align: center;
                line-height: 24px;
                font-size: 12px;
                font-weight: 600;
                margin-right: 12px;
            }}
            
            .error-indicator {{
                color: #d83b01;
                font-weight: 600;
                margin-right: 8px;
            }}
            
            /* Custom scrollbar */
            .modal-body::-webkit-scrollbar {{ 
                width: 8px; 
            }}
            .modal-body::-webkit-scrollbar-track {{ 
                background: #f1f1f1; 
                border-radius: 4px;
            }}
            .modal-body::-webkit-scrollbar-thumb {{ 
                background: #c8c6c4; 
                border-radius: 4px; 
            }}
            .modal-body::-webkit-scrollbar-thumb:hover {{ 
                background: #a19f9d; 
            }}
            
            /* Responsive Design */
            @media (max-width: 768px) {{
                .container {{
                    padding: 12px;
                }}
                
                .cluster-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .stats-bar {{
                    flex-wrap: wrap;
                    gap: 16px;
                }}
                
                .modal {{
                    width: 95%;
                    margin: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Error Analysis</h1>
                <p>AI-powered root cause analysis and recommended fixes for Security Copilot failures.</p>
            </div>
            <div class="nav">
                <a href="/">Customer Insights</a>
                <a href="/clusters" class="active">Error Analysis</a>
                <a href="/api/recluster" onclick="recluster()">Re-run Analysis</a>
            </div>
            <div class="stats-bar">
                <div class="stat-card">
                    <div class="stat-number">{len(analytics['clusters'])}</div>
                    <div class="stat-label">Error Patterns</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{sum(cluster['size'] for cluster in analytics['clusters'])}</div>
                    <div class="stat-label">Total Failures</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{max(cluster['size'] for cluster in analytics['clusters']) if analytics['clusters'] else 0}</div>
                    <div class="stat-label">Largest Error Group</div>
                </div>
            </div>
            <div class="cluster-grid">
                {chr(10).join([
                    f'''<div class="cluster-card">
                        <div class="cluster-content">
                            <div class="cluster-header">
                                <div class="error-type">{(cluster_summaries.get(cluster['cluster_id']).label or 'Error Pattern') if cluster_summaries.get(cluster['cluster_id']) else 'Error Pattern'}</div>
                                <div class="cluster-size">{cluster['size']} failures</div>
                            </div>
                            <div class="root-cause">
                                <strong>Root Cause Analysis:</strong><br>
                                {cluster['root_cause'][:200]}{'...' if len(cluster['root_cause']) > 200 else ''}
                            </div>
                            <div class="recommendations">
                                <strong>Recommended Fixes:</strong><br>
                                {'<br>• '.join([''] + cluster.get('recommendations', ['No recommendations available'])[:3])}
                            </div>
                            <button class="expand-btn" onclick="openModal('modal-{cluster['cluster_id']}')">
                                <span class="loading" style="display: none;"></span>
                                View Failed Prompts
                            </button>
                        </div>
                    </div>''' for cluster in analytics['clusters']])}
            </div>
        </div>
        
        <!-- Modals for each cluster -->
        {chr(10).join([
            f'''<div class="modal-overlay" id="overlay-{cluster['cluster_id']}">
                <div class="modal" id="modal-{cluster['cluster_id']}">
                    <div class="modal-header">
                        <h3 class="modal-title">Failed Prompts - {(cluster_summaries.get(cluster['cluster_id']).label or 'Error Pattern') if cluster_summaries.get(cluster['cluster_id']) else 'Error Pattern'}</h3>
                        <button class="modal-close" onclick="closeModal('modal-{cluster['cluster_id']}')">&times;</button>
                    </div>
                    <div class="modal-body">
                        {chr(10).join([f'<div class="prompt-item-modal"><span class="prompt-number">{i+1}</span><span class="error-indicator">ERROR:</span>"{p}"</div>' for i, p in enumerate(cluster_prompts.get(cluster['cluster_id'], [])[:15])])}
                    </div>
                </div>
            </div>''' for cluster in analytics['clusters']])}
        
        <script>
            function openModal(modalId) {{
                const overlay = document.getElementById('overlay-' + modalId.split('-')[1]);
                const modal = document.getElementById(modalId);
                
                overlay.classList.add('active');
                modal.classList.add('active');
                
                // Prevent body scroll
                document.body.style.overflow = 'hidden';
            }}
            
            function closeModal(modalId) {{
                const overlay = document.getElementById('overlay-' + modalId.split('-')[1]);
                const modal = document.getElementById(modalId);
                
                overlay.classList.remove('active');
                modal.classList.remove('active');
                
                // Restore body scroll
                document.body.style.overflow = 'auto';
            }}
            
            // Close modal when clicking overlay
            document.addEventListener('click', function(e) {{
                if (e.target.classList.contains('modal-overlay')) {{
                    const modalId = e.target.querySelector('.modal').id;
                    closeModal(modalId);
                }}
            }});
            
            // Close modal with Escape key
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') {{
                    const activeModal = document.querySelector('.modal-overlay.active');
                    if (activeModal) {{
                        const modalId = activeModal.querySelector('.modal').id;
                        closeModal(modalId);
                    }}
                }}
            }});
            
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
