from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from database import get_db, create_tables
from models import TelemetryEvent, Cluster
from ingest import telemetry_ingestor
from embeddings import embedding_generator
from clustering import cluster_manager
from summarization import cluster_summarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Copilot Failure Analysis API",
    description="API for ingesting, clustering, and analyzing failed Copilot responses",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class TelemetryEventCreate(BaseModel):
    evaluation_id: str
    customer_prompt: str
    skill_input: str
    skill_output: Dict[str, Any]
    ai_output: Optional[str] = None

class TelemetryEventBatch(BaseModel):
    events: List[TelemetryEventCreate]

class ClusterResponse(BaseModel):
    cluster_id: int
    size: int
    algorithm: str
    summary: Optional[str]
    root_cause: Optional[str]
    common_plugins: List[str]
    common_endpoints: List[str]
    common_error_codes: List[int]
    sample_prompts: List[str]

class ClusteringRequest(BaseModel):
    algorithm: str = "hdbscan"
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    n_clusters: Optional[int] = None

class ProcessingStatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Ingest endpoints
@app.post("/ingest/event", response_model=ProcessingStatusResponse)
async def ingest_single_event(event: TelemetryEventCreate):
    """Ingest a single telemetry event"""
    try:
        success = telemetry_ingestor.ingest_event(event.dict())
        if success:
            return ProcessingStatusResponse(
                status="success",
                message="Event ingested successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to ingest event")
    except Exception as e:
        logger.error(f"Error ingesting event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/batch", response_model=ProcessingStatusResponse)
async def ingest_batch_events(batch: TelemetryEventBatch):
    """Ingest a batch of telemetry events"""
    try:
        events_data = [event.dict() for event in batch.events]
        success_count = telemetry_ingestor.ingest_batch(events_data)
        
        return ProcessingStatusResponse(
            status="success",
            message=f"Batch processing completed",
            details={
                "total_events": len(batch.events),
                "successful_ingests": success_count,
                "failed_ingests": len(batch.events) - success_count
            }
        )
    except Exception as e:
        logger.error(f"Error ingesting batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Embedding endpoints
@app.post("/embeddings/generate", response_model=ProcessingStatusResponse)
async def generate_embeddings(
    background_tasks: BackgroundTasks,
    source: str = "customer_prompt+skill_output",
    use_openai: bool = True
):
    """Generate embeddings for all events (background task)"""
    try:
        # Get all event IDs without embeddings
        from database import get_db_session
        from sqlalchemy.orm import Session
        
        db = get_db_session()
        try:
            event_ids = db.query(TelemetryEvent.id).outerjoin(
                embedding_generator.Embedding
            ).filter(
                embedding_generator.Embedding.id.is_(None)
            ).all()
            event_ids = [id[0] for id in event_ids]
        finally:
            db.close()
        
        if not event_ids:
            return ProcessingStatusResponse(
                status="success",
                message="No events require embedding generation"
            )
        
        # Start background task
        background_tasks.add_task(
            embedding_generator.generate_embeddings_batch,
            event_ids, source, use_openai
        )
        
        return ProcessingStatusResponse(
            status="processing",
            message="Embedding generation started in background",
            details={"event_count": len(event_ids)}
        )
        
    except Exception as e:
        logger.error(f"Error starting embedding generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Clustering endpoints
@app.post("/clustering/perform", response_model=ProcessingStatusResponse)
async def perform_clustering(
    background_tasks: BackgroundTasks,
    request: ClusteringRequest
):
    """Perform clustering on embeddings (background task)"""
    try:
        # Prepare clustering parameters
        kwargs = {}
        if request.min_cluster_size:
            kwargs['min_cluster_size'] = request.min_cluster_size
        if request.min_samples:
            kwargs['min_samples'] = request.min_samples
        if request.n_clusters:
            kwargs['n_clusters'] = request.n_clusters
        
        # Start background task
        background_tasks.add_task(
            cluster_manager.perform_clustering,
            request.algorithm, **kwargs
        )
        
        return ProcessingStatusResponse(
            status="processing",
            message="Clustering started in background",
            details={"algorithm": request.algorithm, "parameters": kwargs}
        )
        
    except Exception as e:
        logger.error(f"Error starting clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clustering/stats", response_model=Dict[str, Any])
async def get_clustering_stats():
    """Get clustering statistics"""
    try:
        stats = cluster_manager.get_cluster_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting clustering stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Summarization endpoints
@app.post("/summarization/generate", response_model=ProcessingStatusResponse)
async def generate_summaries(background_tasks: BackgroundTasks):
    """Generate summaries for all clusters (background task)"""
    try:
        # Start background task
        background_tasks.add_task(cluster_summarizer.summarize_all_clusters)
        
        return ProcessingStatusResponse(
            status="processing",
            message="Summary generation started in background"
        )
        
    except Exception as e:
        logger.error(f"Error starting summary generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarization/cluster/{cluster_id}", response_model=ProcessingStatusResponse)
async def generate_cluster_summary(cluster_id: int, background_tasks: BackgroundTasks):
    """Generate summary for a specific cluster"""
    try:
        # Start background task
        background_tasks.add_task(cluster_summarizer.summarize_cluster, cluster_id)
        
        return ProcessingStatusResponse(
            status="processing",
            message=f"Summary generation started for cluster {cluster_id}"
        )
        
    except Exception as e:
        logger.error(f"Error starting cluster summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main endpoint - Top clusters with summaries
@app.get("/clusters/top", response_model=List[ClusterResponse])
async def get_top_clusters(limit: int = 10):
    """Get top clusters by size with their summaries"""
    try:
        clusters = cluster_summarizer.get_top_clusters(limit)
        return [ClusterResponse(**cluster) for cluster in clusters]
        
    except Exception as e:
        logger.error(f"Error getting top clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Individual cluster endpoint
@app.get("/clusters/{cluster_id}", response_model=ClusterResponse)
async def get_cluster(cluster_id: int):
    """Get detailed information about a specific cluster"""
    try:
        clusters = cluster_summarizer.get_top_clusters(1000)  # Get all clusters
        cluster = next((c for c in clusters if c['cluster_id'] == cluster_id), None)
        
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")
        
        return ClusterResponse(**cluster)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cluster {cluster_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Pipeline endpoint - Full processing pipeline
@app.post("/pipeline/process", response_model=ProcessingStatusResponse)
async def process_pipeline(
    background_tasks: BackgroundTasks,
    clustering_request: ClusteringRequest,
    source: str = "customer_prompt+skill_output",
    use_openai: bool = True
):
    """Run the full processing pipeline: embeddings -> clustering -> summarization"""
    try:
        # Get event IDs without embeddings
        from database import get_db_session
        
        db = get_db_session()
        try:
            event_ids = db.query(TelemetryEvent.id).outerjoin(
                embedding_generator.Embedding
            ).filter(
                embedding_generator.Embedding.id.is_(None)
            ).all()
            event_ids = [id[0] for id in event_ids]
        finally:
            db.close()
        
        # Define pipeline function
        async def run_pipeline():
            """Run the complete pipeline"""
            logger.info("Starting pipeline execution")
            
            # Step 1: Generate embeddings
            if event_ids:
                logger.info(f"Generating embeddings for {len(event_ids)} events")
                embedding_generator.generate_embeddings_batch(event_ids, source, use_openai)
            
            # Step 2: Perform clustering
            logger.info("Performing clustering")
            kwargs = {}
            if clustering_request.min_cluster_size:
                kwargs['min_cluster_size'] = clustering_request.min_cluster_size
            if clustering_request.min_samples:
                kwargs['min_samples'] = clustering_request.min_samples
            if clustering_request.n_clusters:
                kwargs['n_clusters'] = clustering_request.n_clusters
            
            cluster_manager.perform_clustering(clustering_request.algorithm, **kwargs)
            
            # Step 3: Generate summaries
            logger.info("Generating cluster summaries")
            cluster_summarizer.summarize_all_clusters()
            
            logger.info("Pipeline execution completed")
        
        # Start background task
        background_tasks.add_task(run_pipeline)
        
        return ProcessingStatusResponse(
            status="processing",
            message="Full pipeline started in background",
            details={
                "events_to_embed": len(event_ids),
                "clustering_algorithm": clustering_request.algorithm
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoint
@app.get("/stats/overview", response_model=Dict[str, Any])
async def get_overview_stats():
    """Get overview statistics"""
    try:
        from database import get_db_session
        from models import TelemetryEvent, Embedding, Cluster, ClusterSummary
        
        db = get_db_session()
        try:
            total_events = db.query(TelemetryEvent).count()
            total_embeddings = db.query(Embedding).count()
            total_clusters = db.query(Cluster).filter(Cluster.is_noise == False).count()
            noise_clusters = db.query(Cluster).filter(Cluster.is_noise == True).count()
            total_summaries = db.query(ClusterSummary).count()
            
            return {
                "total_events": total_events,
                "total_embeddings": total_embeddings,
                "total_clusters": total_clusters,
                "noise_clusters": noise_clusters,
                "total_summaries": total_summaries,
                "embedding_coverage": f"{(total_embeddings/total_events*100):.1f}%" if total_events > 0 else "0%",
                "summary_coverage": f"{(total_summaries/total_clusters*100):.1f}%" if total_clusters > 0 else "0%"
            }
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting overview stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    from config import config
    
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT) 