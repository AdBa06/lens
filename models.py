from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"
    
    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(String, unique=True, index=True)
    customer_prompt = Column(Text)
    skill_input = Column(Text)
    skill_output = Column(JSON)
    ai_output = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to failure fingerprint
    fingerprint = relationship("FailureFingerprint", back_populates="event", uselist=False)
    embedding = relationship("Embedding", back_populates="event", uselist=False)

class FailureFingerprint(Base):
    __tablename__ = "failure_fingerprints"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("telemetry_events.id"))
    
    # Extracted error metadata
    plugin_name = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)
    status_code = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    error_type = Column(String, nullable=True)
    
    # Generated fingerprint
    fingerprint_hash = Column(String, index=True)
    fingerprint_data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    event = relationship("TelemetryEvent", back_populates="fingerprint")

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("telemetry_events.id"))
    
    # Embedding vectors (stored as JSON array)
    embedding_vector = Column(JSON)
    embedding_model = Column(String)
    embedding_source = Column(String)  # "customer_prompt+skill_output" or "skill_input+skill_output"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    event = relationship("TelemetryEvent", back_populates="embedding")
    cluster_assignment = relationship("ClusterAssignment", back_populates="embedding", uselist=False)

class Cluster(Base):
    __tablename__ = "clusters"
    
    id = Column(Integer, primary_key=True, index=True)
    cluster_label = Column(Integer, index=True)  # -1 for noise points
    cluster_algorithm = Column(String)  # "hdbscan" or "kmeans"
    cluster_parameters = Column(JSON)
    
    # Cluster statistics
    size = Column(Integer)
    is_noise = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assignments = relationship("ClusterAssignment", back_populates="cluster")
    summary = relationship("ClusterSummary", back_populates="cluster", uselist=False)

class ClusterAssignment(Base):
    __tablename__ = "cluster_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    embedding_id = Column(Integer, ForeignKey("embeddings.id"))
    cluster_id = Column(Integer, ForeignKey("clusters.id"))
    
    # Distance/confidence metrics
    distance_to_centroid = Column(Float, nullable=True)
    membership_probability = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    embedding = relationship("Embedding", back_populates="cluster_assignment")
    cluster = relationship("Cluster", back_populates="assignments")

class ClusterSummary(Base):
    __tablename__ = "cluster_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"))
    
    # GPT-4 generated summary
    label = Column(String(255))  # Short, business-meaningful label
    summary_text = Column(Text)
    root_cause = Column(Text)
    recommendations = Column(JSON)  # List of actionable recommendations
    
    # Common patterns
    common_plugins = Column(JSON)
    common_endpoints = Column(JSON)
    common_error_codes = Column(JSON)
    sample_prompts = Column(JSON)
    
    # Metadata
    gpt_model = Column(String)
    generated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    cluster = relationship("Cluster", back_populates="summary") 