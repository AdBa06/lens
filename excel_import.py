#!/usr/bin/env python3
"""
Excel import utility for Copilot Failure Analysis System
Converts Excel sheets to telemetry events
"""

import pandas as pd
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExcelImporter:
    """Import telemetry data from Excel files"""
    
    def __init__(self):
        self.required_columns = [
            'evaluation_id',
            'customer_prompt', 
            'skill_input',
            'skill_output'
        ]
        
        # Column mapping - maps Excel columns to our expected names
        self.column_mapping = {
            # Common variations
            'id': 'evaluation_id',
            'eval_id': 'evaluation_id',
            'event_id': 'evaluation_id',
            'prompt': 'customer_prompt',
            'user_prompt': 'customer_prompt',
            'input': 'skill_input',
            'skill_response': 'skill_output',
            'output': 'skill_output',
            'error_output': 'skill_output',
            'ai_response': 'ai_output',
            'response': 'ai_output'
        }
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect and map columns from DataFrame"""
        detected = {}
        df_columns = [col.lower().strip() for col in df.columns]
        
        # Try exact matches first
        for required in self.required_columns:
            if required in df_columns:
                detected[required] = df.columns[df_columns.index(required)]
                continue
        
        # Try mapped variations
        for excel_col, our_col in self.column_mapping.items():
            if excel_col in df_columns and our_col not in detected:
                detected[our_col] = df.columns[df_columns.index(excel_col)]
        
        # Try partial matches
        for required in self.required_columns:
            if required not in detected:
                for col in df.columns:
                    if required.replace('_', '').lower() in col.lower().replace('_', ''):
                        detected[required] = col
                        break
        
        return detected
    
    def parse_skill_output(self, value: Any) -> Dict[str, Any]:
        """Parse skill_output field - could be JSON string or structured data"""
        if pd.isna(value) or value == '':
            return {"error": "Unknown", "message": "No output provided"}
        
        # If it's already a dict, return as-is
        if isinstance(value, dict):
            return value
        
        # Try to parse as JSON
        try:
            if isinstance(value, str):
                return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # If it's a simple string, create a basic error structure
        error_text = str(value)
        
        # Try to extract common patterns
        skill_output = {
            "error": "Unknown",
            "message": error_text,
            "status_code": None,
            "plugin": None,
            "endpoint": None
        }
        
        # Extract status codes
        import re
        status_match = re.search(r'\b(40[0-9]|50[0-9])\b', error_text)
        if status_match:
            skill_output["status_code"] = int(status_match.group(1))
        
        # Extract common error patterns
        if 'forbidden' in error_text.lower() or 'permission' in error_text.lower():
            skill_output["error"] = "GraphAPI.Forbidden"
            skill_output["status_code"] = skill_output.get("status_code", 403)
        elif 'not found' in error_text.lower():
            skill_output["error"] = "GraphAPI.NotFound"
            skill_output["status_code"] = skill_output.get("status_code", 404)
        elif 'unauthorized' in error_text.lower() or 'auth' in error_text.lower():
            skill_output["error"] = "GraphAPI.Unauthorized"
            skill_output["status_code"] = skill_output.get("status_code", 401)
        elif 'timeout' in error_text.lower():
            skill_output["error"] = "NetworkTimeout"
            skill_output["status_code"] = skill_output.get("status_code", 408)
        
        # Extract plugin names
        plugins = ['calendar', 'outlook', 'teams', 'sharepoint', 'onedrive', 'powerbi', 'planner']
        for plugin in plugins:
            if plugin in error_text.lower():
                skill_output["plugin"] = plugin.title()
                break
        
        return skill_output
    
    def import_from_excel(self, file_path: str, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Import telemetry data from Excel file"""
        try:
            logger.info(f"Reading Excel file: {file_path}")
            
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            logger.info(f"Found {len(df)} rows in Excel file")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Detect column mapping
            column_map = self.detect_columns(df)
            logger.info(f"Detected column mapping: {column_map}")
            
            # Check for required columns
            missing = [col for col in self.required_columns if col not in column_map]
            if missing:
                logger.warning(f"Missing required columns: {missing}")
                # Try to provide defaults for some missing columns
                if 'evaluation_id' not in column_map:
                    logger.info("Generating evaluation IDs from row numbers")
                    df['generated_eval_id'] = [f"excel_row_{i+1:04d}" for i in range(len(df))]
                    column_map['evaluation_id'] = 'generated_eval_id'
                
                if 'skill_input' not in column_map:
                    logger.info("Using generic skill_input")
                    df['generated_skill_input'] = "User request processing"
                    column_map['skill_input'] = 'generated_skill_input'
            
            # Convert to telemetry events
            events = []
            for idx, row in df.iterrows():
                try:
                    # Extract required fields
                    event = {}
                    
                    for our_col, excel_col in column_map.items():
                        if excel_col in df.columns:
                            value = row[excel_col]
                            if our_col == 'skill_output':
                                event[our_col] = self.parse_skill_output(value)
                            else:
                                event[our_col] = str(value) if not pd.isna(value) else ""
                    
                    # Add optional ai_output if available
                    if 'ai_output' in column_map:
                        ai_value = row[column_map['ai_output']]
                        event['ai_output'] = str(ai_value) if not pd.isna(ai_value) else None
                    
                    # Validate event has minimum required fields
                    if all(field in event for field in ['evaluation_id', 'customer_prompt', 'skill_input', 'skill_output']):
                        events.append(event)
                    else:
                        logger.warning(f"Skipping row {idx+1}: missing required fields")
                
                except Exception as e:
                    logger.error(f"Error processing row {idx+1}: {e}")
                    continue
            
            logger.info(f"Successfully converted {len(events)} rows to telemetry events")
            return events
            
        except Exception as e:
            logger.error(f"Error importing Excel file: {e}")
            raise
    
    def create_sample_excel(self, filename: str = "sample_telemetry_template.xlsx"):
        """Create a sample Excel template for users"""
        # Create sample data
        sample_data = {
            'evaluation_id': [
                'copilot_001', 'copilot_002', 'copilot_003', 'copilot_004', 'copilot_005'
            ],
            'customer_prompt': [
                'Create a meeting for next Monday with the team',
                'Find emails from John about the project',
                'Schedule a Teams call with external partners',
                'Upload the project documents to SharePoint',
                'Show me the sales dashboard'
            ],
            'skill_input': [
                'Create calendar event',
                'Search emails',
                'Schedule Teams call',
                'Upload SharePoint documents',
                'Display PowerBI dashboard'
            ],
            'skill_output': [
                '{"error": "GraphAPI.Forbidden", "status_code": 403, "message": "Insufficient privileges", "plugin": "Calendar", "endpoint": "/me/events"}',
                '{"error": "GraphAPI.NotFound", "status_code": 404, "message": "User not found", "plugin": "Outlook", "endpoint": "/me/messages"}',
                '{"error": "GraphAPI.Unauthorized", "status_code": 401, "message": "Authentication failed", "plugin": "Teams", "endpoint": "/me/teams"}',
                'Error: Access denied for SharePoint operation',
                'PowerBI service timeout - 408 error'
            ],
            'ai_output': [
                "I'm sorry, I don't have permission to create calendar events.",
                "I couldn't find the user you're looking for.",
                "I need authentication to access Teams.",
                "I encountered an error accessing SharePoint.",
                "The PowerBI service is currently unavailable."
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_excel(filename, index=False)
        
        logger.info(f"✅ Created sample Excel template: {filename}")
        logger.info("Column descriptions:")
        logger.info("  - evaluation_id: Unique identifier for each event")
        logger.info("  - customer_prompt: What the user asked Copilot")
        logger.info("  - skill_input: How Copilot interpreted the request")
        logger.info("  - skill_output: Error details (can be JSON or text)")
        logger.info("  - ai_output: Natural language response (optional)")

def import_and_ingest_excel(file_path: str, sheet_name: str = None) -> bool:
    """Import Excel data and ingest into the system"""
    try:
        # Import Excel data
        importer = ExcelImporter()
        events = importer.import_from_excel(file_path, sheet_name)
        
        if not events:
            logger.error("No events were imported from Excel file")
            return False
        
        # Ingest into system
        from ingest import telemetry_ingestor
        success_count = telemetry_ingestor.ingest_batch(events)
        
        logger.info(f"Successfully ingested {success_count}/{len(events)} events from Excel")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error importing and ingesting Excel data: {e}")
        return False

def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python excel_import.py create_template    # Create sample Excel template")
        print("  python excel_import.py import <file.xlsx>  # Import Excel file")
        print("  python excel_import.py import <file.xlsx> <sheet_name>  # Import specific sheet")
        return
    
    command = sys.argv[1]
    
    if command == "create_template":
        importer = ExcelImporter()
        importer.create_sample_excel()
        
    elif command == "import":
        if len(sys.argv) < 3:
            print("Please provide Excel file path")
            return
        
        file_path = sys.argv[2]
        sheet_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        success = import_and_ingest_excel(file_path, sheet_name)
        if success:
            print("✅ Excel data imported successfully!")
        else:
            print("❌ Failed to import Excel data")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main() 