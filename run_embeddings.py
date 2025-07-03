#!/usr/bin/env python3
"""
Enhanced Embedding Generation Script
Supports multiple embedding models and regeneration options
"""
import argparse
import logging
from embeddings import embedding_generator
from database import get_db_session
from models import TelemetryEvent, Embedding
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for telemetry events')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI embeddings (requires API key)')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate all embeddings (delete existing)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--source', default='customer_prompt+skill_output', 
                       help='Source text for embeddings (customer_prompt+skill_output or skill_input+skill_output)')
    
    args = parser.parse_args()
    
    print("🚀 Starting enhanced embedding generation...")
    print(f"📋 Config: OpenAI={args.openai}, Regenerate={args.regenerate}, Batch Size={args.batch_size}")
    
    # Handle regeneration
    if args.regenerate:
        print("🔄 Regenerating all embeddings...")
        db = get_db_session()
        try:
            deleted_count = db.query(Embedding).delete()
            db.commit()
            print(f"🗑️ Deleted {deleted_count} existing embeddings")
        finally:
            db.close()
    
    # Get events that need embeddings
    db = get_db_session()
    try:
        query = db.query(TelemetryEvent.id).outerjoin(Embedding).filter(Embedding.id.is_(None))
        event_ids = [id[0] for id in query.all()]
        print(f"📊 Found {len(event_ids)} events needing embeddings")
    finally:
        db.close()
    
    if not event_ids:
        print("✅ No events need embeddings")
        return
    
    # Validate OpenAI configuration if requested
    if args.openai and not config.OPENAI_API_KEY:
        print("❌ OpenAI API key not configured. Falling back to sentence transformers.")
        args.openai = False
    
    # Generate embeddings
    print(f"🧠 Generating embeddings using {'OpenAI' if args.openai else 'Sentence Transformers'}...")
    result = embedding_generator.generate_embeddings_batch(
        event_ids, 
        source=args.source,
        use_openai=args.openai
    )
    print(f"🎉 Generated {result} embeddings successfully!")
    
    # Summary
    db = get_db_session()
    try:
        total_events = db.query(TelemetryEvent).count()
        total_embeddings = db.query(Embedding).count()
        print(f"📈 Summary: {total_embeddings}/{total_events} events have embeddings")
    finally:
        db.close()

if __name__ == "__main__":
    main() 