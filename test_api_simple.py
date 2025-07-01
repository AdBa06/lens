#!/usr/bin/env python3
"""
Simple API test for Copilot Failure Analysis System
Tests API endpoints without heavy ML dependencies
"""

import json
import logging
import time
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_api():
    """Create a simplified version of the API for testing"""
    try:
        # Check if FastAPI is available
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        except ImportError:
            logger.error("FastAPI not installed. Installing with pip...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fastapi', 'pydantic'])
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        
        # Import our components
        from database import create_tables, get_db_session
        from models import TelemetryEvent
        from ingest import telemetry_ingestor
        
        app = FastAPI(title="Copilot Failure Analysis - Simple Test API")
        
        class TelemetryEventCreate(BaseModel):
            evaluation_id: str
            customer_prompt: str
            skill_input: str
            skill_output: Dict[str, Any]
            ai_output: str = None
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time()}
        
        @app.post("/ingest/event")
        async def ingest_event(event: TelemetryEventCreate):
            try:
                success = telemetry_ingestor.ingest_event(event.dict())
                if success:
                    return {"status": "success", "message": "Event ingested successfully"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to ingest event")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/stats/simple")
        async def get_simple_stats():
            try:
                db = get_db_session()
                try:
                    total_events = db.query(TelemetryEvent).count()
                    return {
                        "total_events": total_events,
                        "database_status": "connected"
                    }
                finally:
                    db.close()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
        
    except Exception as e:
        logger.error(f"Error creating API: {e}")
        return None

def test_api_endpoints():
    """Test API endpoints manually"""
    logger.info("Testing API endpoint logic...")
    
    try:
        # Test ingestion directly
        from ingest import telemetry_ingestor
        
        # Create test event
        test_event = {
            "evaluation_id": "api_test_001",
            "customer_prompt": "Schedule a meeting with the team",
            "skill_input": "Create calendar event",
            "skill_output": {
                "error": "GraphAPI.Forbidden",
                "status_code": 403,
                "message": "Insufficient privileges",
                "plugin": "Calendar",
                "endpoint": "/me/events"
            },
            "ai_output": "I cannot create calendar events due to permissions."
        }
        
        # Test ingestion
        success = telemetry_ingestor.ingest_event(test_event)
        if success:
            logger.info("✅ API ingestion endpoint logic - PASSED")
        else:
            logger.error("❌ API ingestion endpoint logic - FAILED")
            return False
        
        # Test stats
        from database import get_db_session
        from models import TelemetryEvent
        
        db = get_db_session()
        try:
            count = db.query(TelemetryEvent).count()
            logger.info(f"✅ API stats endpoint logic - PASSED (found {count} events)")
        finally:
            db.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API endpoint test failed: {e}")
        return False

def test_with_curl_commands():
    """Generate curl commands for testing"""
    logger.info("Generating curl test commands...")
    
    # Sample curl commands
    curl_commands = [
        {
            "name": "Health Check",
            "command": 'curl -X GET "http://localhost:8000/health"'
        },
        {
            "name": "Ingest Event",
            "command": '''curl -X POST "http://localhost:8000/ingest/event" \\
  -H "Content-Type: application/json" \\
  -d '{
    "evaluation_id": "curl_test_001",
    "customer_prompt": "Create a meeting for Monday",
    "skill_input": "Create calendar event",
    "skill_output": {
      "error": "GraphAPI.Forbidden",
      "status_code": 403,
      "message": "Insufficient privileges",
      "plugin": "Calendar",
      "endpoint": "/me/events"
    },
    "ai_output": "I cannot create calendar events."
  }'\"\"\"'''
        },
        {
            "name": "Get Stats",
            "command": 'curl -X GET "http://localhost:8000/stats/simple"'
        }
    ]
    
    logger.info("📋 Manual API Testing Commands:")
    logger.info("=" * 50)
    
    for cmd in curl_commands:
        logger.info(f"\n🔹 {cmd['name']}:")
        logger.info(f"   {cmd['command']}")
    
    logger.info("\n" + "=" * 50)
    logger.info("💡 To test the API:")
    logger.info("1. Start the API: python -c \"from test_api_simple import create_simple_api; import uvicorn; app=create_simple_api(); uvicorn.run(app, host='127.0.0.1', port=8000)\"")
    logger.info("2. In another terminal, run the curl commands above")
    logger.info("3. Or visit http://localhost:8000/docs for interactive API docs")

def run_simple_api_server():
    """Run the simplified API server"""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'uvicorn'])
        import uvicorn
    
    # Initialize database
    from database import create_tables
    create_tables()
    
    app = create_simple_api()
    if app:
        logger.info("🚀 Starting simplified API server on http://localhost:8000")
        logger.info("📖 API docs available at http://localhost:8000/docs")
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        logger.error("Failed to create API")

def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("COPILOT FAILURE ANALYSIS - API TESTS")
    logger.info("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_simple_api_server()
        return
    
    # Run endpoint logic tests
    logger.info("\n--- Testing API Endpoint Logic ---")
    api_success = test_api_endpoints()
    
    # Show curl commands
    logger.info("\n--- Manual Testing Commands ---")
    test_with_curl_commands()
    
    logger.info("\n" + "=" * 60)
    if api_success:
        logger.info("✅ API endpoint logic tests passed!")
        logger.info("\n🎯 Next Steps:")
        logger.info("1. Run: python test_api_simple.py --server")
        logger.info("2. Test endpoints with curl commands shown above")
        logger.info("3. Or install full dependencies and run: python main.py --sample")
    else:
        logger.error("❌ Some API tests failed")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 