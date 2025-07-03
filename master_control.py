#!/usr/bin/env python3
"""
Master Control Script for Enhanced Copilot Failure Analysis
Orchestrates the complete ML + LLM pipeline with business intelligence
"""

import argparse
import logging
import sys
from datetime import datetime
from database import get_db_session, create_tables
from models import TelemetryEvent, Embedding, Cluster, ClusterSummary
from embeddings import embedding_generator
from enhanced_clustering import enhanced_cluster_manager
from business_llm_validator import business_llm_validator
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterController:
    """Master controller for the enhanced analysis pipeline"""
    
    def __init__(self):
        self.db = None
    
    def initialize_system(self):
        """Initialize the complete system"""
        print("🚀 Initializing Enhanced Copilot Failure Analysis System...")
        
        try:
            # Create database tables
            create_tables()
            print("✅ Database tables initialized")
            
            # Check data availability
            db = get_db_session()
            try:
                event_count = db.query(TelemetryEvent).count()
                embedding_count = db.query(Embedding).count()
                cluster_count = db.query(Cluster).count()
                
                print(f"📊 System Status:")
                print(f"   - Events: {event_count}")
                print(f"   - Embeddings: {embedding_count}")
                print(f"   - Clusters: {cluster_count}")
                
                return event_count > 0
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def run_complete_pipeline(self, use_openai: bool = False, regenerate_embeddings: bool = False):
        """Run the complete ML + LLM pipeline"""
        print("🎯 Running Complete Enhanced Pipeline...")
        
        try:
            # Step 1: Generate/Update Embeddings
            print("\n📊 Step 1: Embedding Generation")
            self._run_embedding_generation(use_openai, regenerate_embeddings)
            
            # Step 2: Enhanced Clustering
            print("\n🧠 Step 2: Enhanced ML Clustering")
            clustering_result = self._run_enhanced_clustering()
            
            if not clustering_result.get("success"):
                print(f"❌ Clustering failed: {clustering_result.get('error')}")
                return False
            
            # Step 3: Business LLM Validation
            print("\n🤖 Step 3: Business LLM Validation")
            self._run_business_validation()
            
            print("\n🎉 Complete pipeline finished successfully!")
            self._print_pipeline_summary()
            return True
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            print(f"❌ Pipeline failed: {e}")
            return False
    
    def _run_embedding_generation(self, use_openai: bool, regenerate: bool):
        """Run embedding generation step"""
        try:
            db = get_db_session()
            
            if regenerate:
                print("🔄 Regenerating all embeddings...")
                deleted_count = db.query(Embedding).delete()
                db.commit()
                print(f"🗑️ Deleted {deleted_count} existing embeddings")
            
            # Get events needing embeddings
            query = db.query(TelemetryEvent.id).outerjoin(Embedding).filter(Embedding.id.is_(None))
            event_ids = [id[0] for id in query.all()]
            db.close()
            
            if not event_ids:
                print("✅ All events already have embeddings")
                return
            
            print(f"📊 Generating embeddings for {len(event_ids)} events...")
            result = embedding_generator.generate_embeddings_batch(
                event_ids, 
                use_openai=use_openai
            )
            print(f"✅ Generated {result} embeddings successfully")
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _run_enhanced_clustering(self):
        """Run enhanced clustering step"""
        try:
            print("🎯 Running enhanced clustering with business intelligence...")
            
            result = enhanced_cluster_manager.perform_enhanced_clustering(
                use_business_weights=True
            )
            
            if result.get("success"):
                print(f"✅ Clustering completed:")
                print(f"   - Method: {result['method']}")
                print(f"   - Clusters: {result['num_clusters']}")
                print(f"   - Silhouette Score: {result['silhouette_score']:.3f}")
                print(f"   - Business Weighted: {result['business_weighted']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced clustering failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_business_validation(self):
        """Run business LLM validation step"""
        try:
            db = get_db_session()
            try:
                # Get all non-noise clusters
                clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
                
                if not clusters:
                    print("⚠️ No clusters found for validation")
                    return
                
                print(f"🤖 Validating {len(clusters)} clusters with business LLM...")
                
                success_count = 0
                for cluster in clusters:
                    try:
                        print(f"   Analyzing cluster {cluster.id} (size: {cluster.size})...")
                        result = business_llm_validator.validate_and_enhance_cluster(cluster.id)
                        
                        if "error" not in result:
                            success_count += 1
                            print(f"   ✅ Cluster {cluster.id}: {result['cluster_metadata']['business_priority']} priority")
                        else:
                            print(f"   ⚠️ Cluster {cluster.id}: {result['error']}")
                            
                    except Exception as e:
                        logger.error(f"Failed to validate cluster {cluster.id}: {e}")
                
                print(f"✅ Business validation completed: {success_count}/{len(clusters)} clusters analyzed")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Business validation failed: {e}")
            raise
    
    def _print_pipeline_summary(self):
        """Print pipeline execution summary"""
        try:
            db = get_db_session()
            try:
                # Get final statistics
                event_count = db.query(TelemetryEvent).count()
                embedding_count = db.query(Embedding).count()
                cluster_count = db.query(Cluster).filter(Cluster.is_noise.is_(False)).count()
                summary_count = db.query(ClusterSummary).count()
                
                print("\n📈 Pipeline Execution Summary:")
                print("=" * 50)
                print(f"📊 Total Events Processed: {event_count}")
                print(f"🧠 Embeddings Generated: {embedding_count}")
                print(f"🎯 Clusters Identified: {cluster_count}")
                print(f"🤖 Business Summaries: {summary_count}")
                print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)
                
                # Get top clusters by business priority
                clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).order_by(Cluster.size.desc()).limit(5).all()
                
                if clusters:
                    print("\n🏆 Top 5 Largest Clusters:")
                    for i, cluster in enumerate(clusters, 1):
                        summary = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == cluster.id).first()
                        priority = "Unknown"
                        if summary:
                            # Try to extract priority from summary
                            summary_text = summary.summary_text.lower()
                            if "critical" in summary_text:
                                priority = "Critical"
                            elif "high" in summary_text:
                                priority = "High"
                            elif "medium" in summary_text:
                                priority = "Medium"
                            else:
                                priority = "Low"
                        
                        print(f"   {i}. Cluster {cluster.id}: {cluster.size} events - {priority} priority")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error printing summary: {e}")
    
    def start_dashboard(self, port: int = 8081):
        """Start the enhanced dashboard"""
        try:
            print(f"🌐 Starting Enhanced Dashboard on port {port}...")
            import subprocess
            subprocess.run([
                sys.executable, "enhanced_dashboard.py"
            ], cwd=".")
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            print(f"❌ Dashboard startup failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Copilot Failure Analysis Master Controller')
    
    parser.add_argument('--init', action='store_true', help='Initialize the system')
    parser.add_argument('--run-pipeline', action='store_true', help='Run the complete ML + LLM pipeline')
    parser.add_argument('--dashboard', action='store_true', help='Start the enhanced dashboard')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI for embeddings (requires API key)')
    parser.add_argument('--regenerate-embeddings', action='store_true', help='Regenerate all embeddings')
    parser.add_argument('--port', type=int, default=8081, help='Dashboard port (default: 8081)')
    
    args = parser.parse_args()
    
    controller = MasterController()
    
    if args.init:
        print("🎬 Initializing Enhanced System...")
        if controller.initialize_system():
            print("✅ System initialization completed successfully!")
        else:
            print("❌ System initialization failed!")
            return 1
    
    if args.run_pipeline:
        print("🚀 Starting Enhanced Pipeline...")
        if not controller.initialize_system():
            print("❌ System not properly initialized!")
            return 1
        
        if controller.run_complete_pipeline(
            use_openai=args.openai,
            regenerate_embeddings=args.regenerate_embeddings
        ):
            print("🎉 Pipeline completed successfully!")
        else:
            print("❌ Pipeline execution failed!")
            return 1
    
    if args.dashboard:
        print("🌐 Starting Enhanced Dashboard...")
        controller.start_dashboard(args.port)
    
    if not any([args.init, args.run_pipeline, args.dashboard]):
        parser.print_help()
        print("\n💡 Quick start suggestions:")
        print("   python master_control.py --init --run-pipeline --dashboard")
        print("   python master_control.py --run-pipeline --openai")
        print("   python master_control.py --dashboard --port 8082")

if __name__ == "__main__":
    exit(main() or 0)
