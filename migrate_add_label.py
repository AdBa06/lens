#!/usr/bin/env python3
"""
Migration script to add 'label' column to cluster_summaries table
"""
import sqlite3
import logging
from database import get_db_session
from models import ClusterSummary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_add_label():
    """Add label column to cluster_summaries table"""
    try:
        # Connect to the database
        db = get_db_session()
        
        # Get the database file path from the connection
        db_path = db.bind.url.database
        db.close()
        
        # Use direct SQLite connection for schema modification
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(cluster_summaries)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'label' not in columns:
            logger.info("Adding 'label' column to cluster_summaries table...")
            cursor.execute("ALTER TABLE cluster_summaries ADD COLUMN label VARCHAR(255)")
            conn.commit()
            logger.info("Successfully added 'label' column")
        else:
            logger.info("Column 'label' already exists in cluster_summaries table")
        
        conn.close()
        
        # Verify the migration worked
        db = get_db_session()
        try:
            # Try to query the new column
            result = db.execute("SELECT label FROM cluster_summaries LIMIT 1").fetchone()
            logger.info("Migration verification successful - 'label' column is accessible")
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_add_label()
    print("✅ Migration completed successfully")
