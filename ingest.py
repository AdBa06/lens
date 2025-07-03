import json
import hashlib
import re
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from models import TelemetryEvent, FailureFingerprint
from database import get_db_session

logger = logging.getLogger(__name__)

class ErrorParser:
    """Parse error information from skill_output"""
    
    def __init__(self):
        self.plugin_patterns = [
            r'"plugin":\s*"([^"]+)"',
            r'"pluginName":\s*"([^"]+)"',
            r'"skill":\s*"([^"]+)"',
            r'plugin[_\s]*name[:\s]*([^\s,}]+)'
        ]
        
        self.endpoint_patterns = [
            r'"endpoint":\s*"([^"]+)"',
            r'"url":\s*"([^"]+)"',
            r'"api[_\s]*endpoint":\s*"([^"]+)"',
            r'https?://[^/]+(/[^\s"]+)'
        ]
        
        self.status_code_patterns = [
            r'"status[_\s]*code":\s*(\d+)',
            r'"statusCode":\s*(\d+)',
            r'"status":\s*(\d+)',
            r'HTTP[_\s]*(\d{3})'
        ]
        
        self.error_message_patterns = [
            r'"error[_\s]*message":\s*"([^"]+)"',
            r'"message":\s*"([^"]+)"',
            r'"error":\s*"([^"]+)"',
            r'"description":\s*"([^"]+)"'
        ]
    
    def extract_plugin_name(self, skill_output: Dict[str, Any]) -> Optional[str]:
        """Extract plugin name from skill_output"""
        output_str = json.dumps(skill_output) if isinstance(skill_output, dict) else str(skill_output)
        
        for pattern in self.plugin_patterns:
            match = re.search(pattern, output_str, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_endpoint(self, skill_output: Dict[str, Any]) -> Optional[str]:
        """Extract API endpoint from skill_output"""
        output_str = json.dumps(skill_output) if isinstance(skill_output, dict) else str(skill_output)
        
        for pattern in self.endpoint_patterns:
            match = re.search(pattern, output_str, re.IGNORECASE)
            if match:
                endpoint = match.group(1).strip()
                # Normalize endpoint by removing query parameters
                if '?' in endpoint:
                    endpoint = endpoint.split('?')[0]
                return endpoint
        
        return None
    
    def extract_status_code(self, skill_output: Dict[str, Any]) -> Optional[int]:
        """Extract HTTP status code from skill_output"""
        output_str = json.dumps(skill_output) if isinstance(skill_output, dict) else str(skill_output)
        
        for pattern in self.status_code_patterns:
            match = re.search(pattern, output_str, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def extract_error_message(self, skill_output: Dict[str, Any]) -> Optional[str]:
        """Extract error message from skill_output"""
        output_str = json.dumps(skill_output) if isinstance(skill_output, dict) else str(skill_output)
        
        for pattern in self.error_message_patterns:
            match = re.search(pattern, output_str, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_error_type(self, skill_output: Dict[str, Any]) -> Optional[str]:
        """Classify error type based on status code and content"""
        status_code = self.extract_status_code(skill_output)
        output_str = json.dumps(skill_output).lower() if isinstance(skill_output, dict) else str(skill_output).lower()
        
        if status_code:
            if 400 <= status_code < 500:
                if 'unauthorized' in output_str or 'forbidden' in output_str:
                    return 'authentication_error'
                elif 'not found' in output_str:
                    return 'resource_not_found'
                elif 'bad request' in output_str:
                    return 'bad_request'
                else:
                    return 'client_error'
            elif 500 <= status_code < 600:
                return 'server_error'
        
        if any(keyword in output_str for keyword in ['timeout', 'timed out']):
            return 'timeout_error'
        elif any(keyword in output_str for keyword in ['network', 'connection']):
            return 'network_error'
        elif any(keyword in output_str for keyword in ['permission', 'access denied']):
            return 'permission_error'
        
        return 'unknown_error'
    
    def parse_error_metadata(self, skill_output: Dict[str, Any]) -> Dict[str, Any]:
        """Parse all error metadata from skill_output"""
        return {
            'plugin_name': self.extract_plugin_name(skill_output),
            'endpoint': self.extract_endpoint(skill_output),
            'status_code': self.extract_status_code(skill_output),
            'error_message': self.extract_error_message(skill_output),
            'error_type': self.extract_error_type(skill_output)
        }

class FingerprintGenerator:
    """Generate failure fingerprints for clustering"""
    
    def __init__(self):
        self.error_parser = ErrorParser()
    
    def normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint for fingerprinting"""
        if not endpoint:
            return ""
        
        # Remove specific IDs and replace with placeholders
        normalized = re.sub(r'/\d+', '/{id}', endpoint)
        normalized = re.sub(r'/[a-fA-F0-9-]{32,}', '/{guid}', normalized)
        normalized = re.sub(r'/[a-fA-F0-9-]{8,}', '/{id}', normalized)
        
        return normalized.lower()
    
    def normalize_error_message(self, error_message: str) -> str:
        """Normalize error message for fingerprinting"""
        if not error_message:
            return ""
        
        # Remove specific values and replace with placeholders
        normalized = re.sub(r'\d+', '{number}', error_message)
        normalized = re.sub(r'[a-fA-F0-9-]{32,}', '{guid}', normalized)
        normalized = re.sub(r'[a-fA-F0-9-]{8,}', '{id}', normalized)
        normalized = re.sub(r'https?://[^\s]+', '{url}', normalized)
        
        return normalized.lower().strip()
    
    def generate_fingerprint(self, skill_output: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate a unique fingerprint for the failure"""
        metadata = self.error_parser.parse_error_metadata(skill_output)
        
        # Create normalized fingerprint components
        fingerprint_components = {
            'plugin_name': metadata.get('plugin_name', '').lower() if metadata.get('plugin_name') else '',
            'endpoint': self.normalize_endpoint(metadata.get('endpoint', '')),
            'status_code': metadata.get('status_code'),
            'error_type': metadata.get('error_type', ''),
            'error_message_normalized': self.normalize_error_message(metadata.get('error_message', ''))
        }
        
        # Create hash from fingerprint components
        fingerprint_string = json.dumps(fingerprint_components, sort_keys=True)
        fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()
        
        return fingerprint_hash, {
            'fingerprint_components': fingerprint_components,
            'original_metadata': metadata
        }

class TelemetryIngestor:
    """Ingest telemetry events and generate fingerprints"""
    
    def __init__(self):
        self.fingerprint_generator = FingerprintGenerator()
        
    def ingest_event(self, event_data: Dict[str, Any]) -> bool:
        """Ingest a single telemetry event"""
        try:
            db = get_db_session()
            
            # Create telemetry event
            telemetry_event = TelemetryEvent(
                evaluation_id=event_data['evaluation_id'],
                customer_prompt=event_data['customer_prompt'],
                skill_input=event_data['skill_input'],
                skill_output=event_data['skill_output'],
                ai_output=event_data.get('ai_output')
            )
            
            db.add(telemetry_event)
            db.flush()  # Get the ID
            
            # Generate failure fingerprint
            fingerprint_hash, fingerprint_data = self.fingerprint_generator.generate_fingerprint(
                event_data['skill_output']
            )
            
            original_metadata = fingerprint_data['original_metadata']
            
            failure_fingerprint = FailureFingerprint(
                event_id=telemetry_event.id,
                plugin_name=original_metadata.get('plugin_name'),
                endpoint=original_metadata.get('endpoint'),
                status_code=original_metadata.get('status_code'),
                error_message=original_metadata.get('error_message'),
                error_type=original_metadata.get('error_type'),
                fingerprint_hash=fingerprint_hash,
                fingerprint_data=fingerprint_data
            )
            
            db.add(failure_fingerprint)
            db.commit()
            
            logger.info(f"Successfully ingested event {event_data['evaluation_id']}")
            db.close()
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting event {event_data.get('evaluation_id', 'unknown')}: {e}")
            if 'db' in locals():
                db.rollback()
                db.close()
            return False
    
    def ingest_batch(self, events: list) -> int:
        """Ingest a batch of telemetry events"""
        success_count = 0
        
        for event_data in events:
            if self.ingest_event(event_data):
                success_count += 1
        
        logger.info(f"Successfully ingested {success_count}/{len(events)} events")
        return success_count

# Singleton instance
telemetry_ingestor = TelemetryIngestor() 