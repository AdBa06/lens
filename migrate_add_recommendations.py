#!/usr/bin/env python3
"""
Migration script to add 'recommendations' column to cluster_summaries table
"""
import sqlite3
import logging
from database import get_db_session

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_add_recommendations():
    """Add recommendations column to cluster_summaries table"""
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
        
        if 'recommendations' not in columns:
            logger.info("Adding 'recommendations' column to cluster_summaries table...")
            cursor.execute("ALTER TABLE cluster_summaries ADD COLUMN recommendations TEXT")
            conn.commit()
            logger.info("Successfully added 'recommendations' column")
        else:
            logger.info("Column 'recommendations' already exists in cluster_summaries table")
        
        conn.close()
        logger.info("Migration completed successfully")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_add_recommendations()
    print("✅ Migration completed successfully")
