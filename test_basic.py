#!/usr/bin/env python3
"""
Basic test script for Copilot Failure Analysis System
Tests core logic without heavy ML dependencies
"""

import json
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_error_parsing():
    """Test the error parsing functionality"""
    logger.info("Testing error parsing...")
    
    # Import our error parser
    try:
        from ingest import ErrorParser
    except ImportError as e:
        logger.error(f"Could not import ErrorParser: {e}")
        return False
    
    parser = ErrorParser()
    
    # Test sample skill outputs
    test_cases = [
        {
            "name": "Calendar Permission Error",
            "skill_output": {
                "error": "GraphAPI.Forbidden",
                "status_code": 403,
                "message": "Insufficient privileges to complete the operation",
                "plugin": "Calendar",
                "endpoint": "/me/events"
            },
            "expected_plugin": "Calendar",
            "expected_endpoint": "/me/events",
            "expected_status": 403,
            "expected_type": "authentication_error"
        },
        {
            "name": "Network Timeout",
            "skill_output": {
                "error": "NetworkTimeout",
                "status_code": 408,
                "message": "Request timed out after 30 seconds",
                "plugin": "Outlook"
            },
            "expected_plugin": "Outlook",
            "expected_status": 408,
            "expected_type": "timeout_error"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        logger.info(f"  Testing: {test_case['name']}")
        
        skill_output = test_case['skill_output']
        metadata = parser.parse_error_metadata(skill_output)
        
        # Check results
        success = True
        
        if metadata['plugin_name'] != test_case.get('expected_plugin'):
            logger.warning(f"    Plugin mismatch: got {metadata['plugin_name']}, expected {test_case.get('expected_plugin')}")
            success = False
            
        if metadata['status_code'] != test_case.get('expected_status'):
            logger.warning(f"    Status code mismatch: got {metadata['status_code']}, expected {test_case.get('expected_status')}")
            success = False
            
        if metadata['error_type'] != test_case.get('expected_type'):
            logger.warning(f"    Error type mismatch: got {metadata['error_type']}, expected {test_case.get('expected_type')}")
            success = False
        
        if success:
            logger.info(f"    ✅ {test_case['name']} - PASSED")
            passed += 1
        else:
            logger.error(f"    ❌ {test_case['name']} - FAILED")
    
    logger.info(f"Error parsing tests: {passed}/{total} passed")
    return passed == total

def test_fingerprint_generation():
    """Test fingerprint generation"""
    logger.info("Testing fingerprint generation...")
    
    try:
        from ingest import FingerprintGenerator
    except ImportError as e:
        logger.error(f"Could not import FingerprintGenerator: {e}")
        return False
    
    generator = FingerprintGenerator()
    
    # Test with identical errors (should have same fingerprint)
    skill_output1 = {
        "error": "GraphAPI.Forbidden",
        "status_code": 403,
        "message": "Access denied for user 12345",
        "plugin": "Calendar",
        "endpoint": "/me/events"
    }
    
    skill_output2 = {
        "error": "GraphAPI.Forbidden", 
        "status_code": 403,
        "message": "Access denied for user 67890",  # Different user ID
        "plugin": "Calendar",
        "endpoint": "/me/events"
    }
    
    fingerprint1, data1 = generator.generate_fingerprint(skill_output1)
    fingerprint2, data2 = generator.generate_fingerprint(skill_output2)
    
    if fingerprint1 == fingerprint2:
        logger.info("    ✅ Identical error patterns produce same fingerprint - PASSED")
        return True
    else:
        logger.error("    ❌ Identical error patterns should produce same fingerprint - FAILED")
        logger.error(f"    Fingerprint 1: {fingerprint1}")
        logger.error(f"    Fingerprint 2: {fingerprint2}")
        return False

def test_database_models():
    """Test database model creation"""
    logger.info("Testing database models...")
    
    try:
        from models import TelemetryEvent, FailureFingerprint
        from database import create_tables, get_db_session
    except ImportError as e:
        logger.error(f"Could not import database components: {e}")
        return False
    
    try:
        # Try to create tables
        create_tables()
        logger.info("    ✅ Database tables created successfully - PASSED")
        
        # Try to create a session
        db = get_db_session()
        db.close()
        logger.info("    ✅ Database session created successfully - PASSED")
        
        return True
        
    except Exception as e:
        logger.error(f"    ❌ Database setup failed: {e}")
        return False

def test_sample_ingestion():
    """Test sample data ingestion"""
    logger.info("Testing sample data ingestion...")
    
    try:
        from ingest import TelemetryIngestor
    except ImportError as e:
        logger.error(f"Could not import TelemetryIngestor: {e}")
        return False
    
    ingestor = TelemetryIngestor()
    
    # Create sample event
    sample_event = {
        "evaluation_id": "test_basic_001",
        "customer_prompt": "Create a test meeting",
        "skill_input": "Create calendar event",
        "skill_output": {
            "error": "GraphAPI.Forbidden",
            "status_code": 403,
            "message": "Test error message",
            "plugin": "Calendar",
            "endpoint": "/me/events"
        },
        "ai_output": "Test AI response"
    }
    
    try:
        success = ingestor.ingest_event(sample_event)
        if success:
            logger.info("    ✅ Sample event ingestion - PASSED")
            return True
        else:
            logger.error("    ❌ Sample event ingestion failed - FAILED")
            return False
    except Exception as e:
        logger.error(f"    ❌ Ingestion error: {e}")
        return False

def run_all_tests():
    """Run all basic tests"""
    logger.info("=" * 60)
    logger.info("COPILOT FAILURE ANALYSIS - BASIC TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Database Models", test_database_models),
        ("Error Parsing", test_error_parsing),
        ("Fingerprint Generation", test_fingerprint_generation),
        ("Sample Ingestion", test_sample_ingestion),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} completed successfully")
            else:
                logger.error(f"❌ {test_name} failed")
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"BASIC TESTS SUMMARY: {passed}/{total} passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 All basic tests passed! Core system is working.")
        logger.info("\nNext steps:")
        logger.info("1. Install ML dependencies if needed: pip install scikit-learn sentence-transformers")
        logger.info("2. Run the full system: python main.py --sample")
        logger.info("3. Test API endpoints with curl or Postman")
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 