import openai
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from models import TelemetryEvent, Embedding
from database import get_db_session
from config import config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for telemetry events"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        self.sentence_transformer = None
        self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """Load sentence transformer model"""
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
    
    def generate_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API"""
        if not self.openai_client:
            logger.warning("OpenAI client not configured")
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            return None
    
    def generate_sentence_transformer_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using sentence transformer"""
        if not self.sentence_transformer:
            logger.warning("Sentence transformer not loaded")
            return None
        
        try:
            embedding = self.sentence_transformer.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating sentence transformer embedding: {e}")
            return None
    
    def prepare_embedding_text(self, event: TelemetryEvent, source: str = "customer_prompt+skill_output") -> str:
        """Prepare text for embedding generation"""
        if source == "customer_prompt+skill_output":
            skill_output_str = str(event.skill_output) if event.skill_output is not None else ""
            return f"Customer Prompt: {event.customer_prompt}\nSkill Output: {skill_output_str}"
        elif source == "skill_input+skill_output":
            skill_output_str = str(event.skill_output) if event.skill_output is not None else ""
            return f"Skill Input: {event.skill_input}\nSkill Output: {skill_output_str}"
        else:
            raise ValueError(f"Invalid embedding source: {source}")
    
    def generate_embedding_for_event(self, event: TelemetryEvent, 
                                   source: str = "customer_prompt+skill_output",
                                   use_openai: bool = True) -> Optional[Embedding]:
        """Generate embedding for a single event"""
        try:
            # Prepare text for embedding
            text = self.prepare_embedding_text(event, source)
            
            # Generate embedding
            if use_openai and self.openai_client:
                embedding_vector = self.generate_openai_embedding(text)
                model_name = config.EMBEDDING_MODEL
            else:
                embedding_vector = self.generate_sentence_transformer_embedding(text)
                model_name = "all-MiniLM-L6-v2"
            
            if embedding_vector is None:
                logger.error(f"Failed to generate embedding for event {event.id}")
                return None
            
            # Create embedding record
            embedding = Embedding(
                event_id=event.id,
                embedding_vector=embedding_vector,
                embedding_model=model_name,
                embedding_source=source
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for event {event.id}: {e}")
            return None
    
    def generate_embeddings_batch(self, event_ids: List[int], 
                                source: str = "customer_prompt+skill_output",
                                use_openai: bool = True) -> int:
        """Generate embeddings for a batch of events"""
        db = get_db_session()
        success_count = 0
        
        try:
            # Get events that don't have embeddings yet
            events = db.query(TelemetryEvent).filter(
                TelemetryEvent.id.in_(event_ids)
            ).outerjoin(Embedding).filter(
                Embedding.id.is_(None)
            ).all()
            
            logger.info(f"Generating embeddings for {len(events)} events")
            
            for event in events:
                embedding = self.generate_embedding_for_event(event, source, use_openai)
                if embedding:
                    db.add(embedding)
                    success_count += 1
                    
                    # Commit in batches to avoid memory issues
                    if success_count % 50 == 0:
                        db.commit()
                        logger.info(f"Generated {success_count} embeddings so far")
            
            db.commit()
            logger.info(f"Successfully generated {success_count} embeddings")
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            db.rollback()
        finally:
            db.close()
        
        return success_count
    
    def get_embeddings_for_clustering(self) -> tuple:
        """Get all embeddings for clustering"""
        db = get_db_session()
        
        try:
            embeddings = db.query(Embedding).all()
            
            if not embeddings:
                logger.warning("No embeddings found for clustering")
                return np.array([]), []
            
            # Extract vectors and IDs
            vectors = []
            embedding_ids = []
            
            for embedding in embeddings:
                vectors.append(embedding.embedding_vector)
                embedding_ids.append(embedding.id)
            
            vectors_array = np.array(vectors)
            logger.info(f"Retrieved {len(vectors)} embeddings for clustering")
            
            return vectors_array, embedding_ids
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings for clustering: {e}")
            return np.array([]), []
        finally:
            db.close()

# Singleton instance
embedding_generator = EmbeddingGenerator() 