#!/usr/bin/env python3
"""
Excel import utility for Error Insights (separate from Customer Analysis)
Converts Excel sheets to telemetry events with unique error-specific IDs
"""

import pandas as pd
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExcelImporterError:
    """Import error insights data from Excel files (ensures unique IDs)"""
    def __init__(self):
        self.required_columns = [
            'evaluation_id',
            'customer_prompt',
            'skill_input',
            'skill_output'
        ]
        self.column_mapping = {
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
        detected = {}
        df_columns = [col.lower().strip() for col in df.columns]
        for required in self.required_columns:
            if required in df_columns:
                detected[required] = df.columns[df_columns.index(required)]
                continue
        for excel_col, our_col in self.column_mapping.items():
            if excel_col in df_columns and our_col not in detected:
                detected[our_col] = df.columns[df_columns.index(excel_col)]
        for required in self.required_columns:
            if required not in detected:
                for col in df.columns:
                    if required.replace('_', '').lower() in col.lower().replace('_', ''):
                        detected[required] = col
                        break
        return detected

    def parse_skill_output(self, value: Any) -> Dict[str, Any]:
        if pd.isna(value) or value == '':
            return {"error": "Unknown", "message": "No output provided"}
        if isinstance(value, dict):
            return value
        try:
            if isinstance(value, str):
                return json.loads(value)
        except json.JSONDecodeError:
            pass
        error_text = str(value)
        skill_output = {
            "error": "Unknown",
            "message": error_text,
            "status_code": None,
            "plugin": None,
            "endpoint": None
        }
        import re
        status_match = re.search(r'\b(40[0-9]|50[0-9])\b', error_text)
        if status_match:
            skill_output["status_code"] = int(status_match.group(1))
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
        plugins = ['calendar', 'outlook', 'teams', 'sharepoint', 'onedrive', 'powerbi', 'planner']
        for plugin in plugins:
            if plugin in error_text.lower():
                skill_output["plugin"] = plugin.title()
                break
        return skill_output

    def import_from_excel(self, file_path: str, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Reading Excel file: {file_path}")
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            logger.info(f"Found {len(df)} rows in Excel file")
            logger.info(f"Columns: {list(df.columns)}")
            column_map = self.detect_columns(df)
            logger.info(f"Detected column mapping: {column_map}")
            missing = [col for col in self.required_columns if col not in column_map]
            if missing:
                logger.warning(f"Missing required columns: {missing}")
                if 'evaluation_id' not in column_map:
                    logger.info("Generating error-specific evaluation IDs from row numbers")
                    df['generated_eval_id'] = [f"error_row_{i+1:04d}" for i in range(len(df))]
                    column_map['evaluation_id'] = 'generated_eval_id'
                if 'skill_input' not in column_map:
                    logger.info("Using default skill_input: NA")
                    df['generated_skill_input'] = "NA"
                    column_map['skill_input'] = 'generated_skill_input'
                if 'skill_output' not in column_map:
                    logger.info("Using default skill_output: NA")
                    df['generated_skill_output'] = '{"error": "NA", "message": "No output data available", "status_code": null}'
                    column_map['skill_output'] = 'generated_skill_output'
            events = []
            for idx, row in df.iterrows():
                try:
                    event = {}
                    for our_col, excel_col in column_map.items():
                        if excel_col in df.columns:
                            value = row[excel_col]
                            if our_col == 'skill_output':
                                event[our_col] = self.parse_skill_output(value)
                            elif our_col == 'evaluation_id':
                                eval_id = str(value) if not pd.isna(value) else ""
                                if not eval_id or eval_id.strip() == "" or eval_id.startswith("synthetic_row_"):
                                    eval_id = f"error_row_{idx+1:04d}"
                                event[our_col] = eval_id
                            else:
                                event[our_col] = str(value) if not pd.isna(value) else ""
                    if 'ai_output' in column_map:
                        ai_value = row[column_map['ai_output']]
                        event['ai_output'] = str(ai_value) if not pd.isna(ai_value) else None
                    events.append(event)
                except Exception as e:
                    logger.error(f"Error processing row {idx+1}: {e}")
                    continue
            logger.info(f"Successfully converted {len(events)} rows to error telemetry events")
            return events
        except Exception as e:
            logger.error(f"Error importing Excel file: {e}")
            raise

def import_and_ingest_excel_error(file_path: str, sheet_name: str = None) -> bool:
    try:
        importer = ExcelImporterError()
        events = importer.import_from_excel(file_path, sheet_name)
        if not events:
            logger.error("No events were imported from Excel file")
            return False
        from ingest import telemetry_ingestor
        # Patch in source_type for error analysis
        for event in events:
            event["source_type"] = "error"
        success_count = telemetry_ingestor.ingest_batch(events)
        logger.info(f"Successfully ingested {success_count}/{len(events)} error events from Excel")
        return success_count > 0
    except Exception as e:
        logger.error(f"Error importing and ingesting Excel error data: {e}")
        return False

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python excel_import_error.py import <file.xlsx>  # Import Error Insights Excel file")
        print("  python excel_import_error.py import <file.xlsx> <sheet_name>  # Import specific sheet")
        return
    command = sys.argv[1]
    if command == "import":
        if len(sys.argv) < 3:
            print("Please provide Excel file path")
            return
        file_path = sys.argv[2]
        sheet_name = sys.argv[3] if len(sys.argv) > 3 else None
        success = import_and_ingest_excel_error(file_path, sheet_name)
        if success:
            print("✅ Error Insights Excel data imported successfully!")
        else:
            print("❌ Failed to import Error Insights Excel data")
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
