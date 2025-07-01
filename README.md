# Copilot Failure Analysis System

A comprehensive backend system for ingesting, clustering, and analyzing failed Copilot responses using telemetry data. The system identifies common failure patterns and provides AI-generated summaries of root causes.

## Features

- **Telemetry Ingestion**: Store and process Copilot failure events
- **Error Parsing**: Extract metadata from skill_output (plugin name, endpoint, status codes, error messages)
- **Failure Fingerprinting**: Generate unique fingerprints for clustering similar failures
- **Vector Embeddings**: Generate embeddings using OpenAI or sentence-transformers
- **Clustering**: Discover failure patterns using HDBSCAN or K-Means
- **AI Summarization**: Generate human-readable summaries using GPT-4
- **REST API**: FastAPI-based API for all operations
- **Modular Architecture**: Clean separation of concerns with comprehensive logging

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telemetry     │    │   Embedding     │    │   Clustering    │
│    Ingestion    │───▶│   Generation    │───▶│    Engine       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Fingerprint   │    │   Vector Store  │    │   Cluster       │
│   Generation    │    │   (Database)    │    │   Analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   SQLite/       │    │   GPT-4         │
                       │   PostgreSQL    │    │   Summaries     │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                └───────────────────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │   REST API      │
                                   │   (FastAPI)     │
                                   └─────────────────┘
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```env
# Database
DATABASE_URL=sqlite:///./copilot_failures.db

# OpenAI (optional, for embeddings and summaries)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002

# Clustering
MIN_CLUSTER_SIZE=5
MIN_SAMPLES=3

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

### 3. Run with Sample Data

```bash
# Initialize system and run with sample data
python main.py --sample

# Use OpenAI for embeddings and summaries (requires API key)
python main.py --sample --openai
```

### 4. Start API Server

```bash
# Using Python
python api.py

# Or using uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Usage

### Ingest Telemetry Events

```bash
# Single event
curl -X POST "http://localhost:8000/ingest/event" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_id": "test_001",
    "customer_prompt": "Create a meeting for Monday",
    "skill_input": "Create calendar event",
    "skill_output": {
      "error": "GraphAPI.Forbidden",
      "status_code": 403,
      "message": "Insufficient privileges",
      "plugin": "Calendar",
      "endpoint": "/me/events"
    },
    "ai_output": "I cannot create calendar events due to permissions."
  }'

# Batch events
curl -X POST "http://localhost:8000/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {...event1...},
      {...event2...}
    ]
  }'
```

### Process Pipeline

```bash
# Full pipeline: embeddings -> clustering -> summarization
curl -X POST "http://localhost:8000/pipeline/process" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "hdbscan",
    "min_cluster_size": 5,
    "min_samples": 3
  }'
```

### Get Results

```bash
# Get top clusters with summaries
curl "http://localhost:8000/clusters/top?limit=10"

# Get specific cluster
curl "http://localhost:8000/clusters/1"

# Get system statistics
curl "http://localhost:8000/stats/overview"
```

## Data Models

### Telemetry Event

```json
{
  "evaluation_id": "unique_identifier",
  "customer_prompt": "what the user asked",
  "skill_input": "how Copilot interpreted the prompt",
  "skill_output": {
    "error": "error_type",
    "status_code": 403,
    "message": "error message",
    "plugin": "plugin_name",
    "endpoint": "/api/endpoint"
  },
  "ai_output": "optional natural language response"
}
```

### Cluster Response

```json
{
  "cluster_id": 1,
  "size": 25,
  "algorithm": "hdbscan",
  "summary": "Calendar permission failures",
  "root_cause": "Users lack necessary permissions for calendar operations",
  "common_plugins": ["Calendar", "Outlook"],
  "common_endpoints": ["/me/events", "/me/calendar"],
  "common_error_codes": [403, 401],
  "sample_prompts": ["Create a meeting...", "Schedule time..."]
}
```

## Component Details

### Ingest Pipeline (`ingest.py`)

- **ErrorParser**: Extracts structured metadata from skill_output
- **FingerprintGenerator**: Creates normalized fingerprints for clustering
- **TelemetryIngestor**: Manages event ingestion and storage

### Embedding Generation (`embeddings.py`)

- **OpenAI Embeddings**: Using text-embedding-ada-002
- **Sentence Transformers**: Local alternative using all-MiniLM-L6-v2
- **Flexible Sources**: customer_prompt+skill_output or skill_input+skill_output

### Clustering (`clustering.py`)

- **HDBSCAN**: Density-based clustering for varying cluster sizes
- **K-Means**: Traditional clustering with automatic K selection
- **Distance Metrics**: Calculate distances to cluster centroids

### Summarization (`summarization.py`)

- **GPT-4 Analysis**: Generate human-readable cluster summaries
- **Pattern Extraction**: Identify common plugins, endpoints, error codes
- **Root Cause Analysis**: AI-generated insights into failure causes

### REST API (`api.py`)

- **FastAPI Framework**: Modern, fast API with automatic documentation
- **Background Tasks**: Long-running operations don't block requests
- **Comprehensive Endpoints**: Full CRUD and processing operations
- **Error Handling**: Detailed error responses and logging

## Development

### Project Structure

```
lens/
├── requirements.txt      # Dependencies
├── config.py            # Configuration management
├── models.py            # SQLAlchemy database models
├── database.py          # Database setup and sessions
├── ingest.py            # Telemetry ingestion pipeline
├── embeddings.py        # Vector embedding generation
├── clustering.py        # Clustering algorithms
├── summarization.py     # GPT-4 summarization
├── api.py              # FastAPI application
├── main.py             # Main application script
└── README.md           # This file
```

### Database Schema

- **telemetry_events**: Raw failure events
- **failure_fingerprints**: Parsed error metadata and fingerprints
- **embeddings**: Vector embeddings for similarity analysis
- **clusters**: Cluster information and metadata
- **cluster_assignments**: Mapping between embeddings and clusters
- **cluster_summaries**: GPT-4 generated analysis

### Logging

All components use structured logging with configurable levels:

```python
logger = logging.getLogger(__name__)
logger.info("Processing started")
logger.error("Error occurred", exc_info=True)
```

## Configuration Options

### Database

- SQLite (default): `sqlite:///./copilot_failures.db`
- PostgreSQL: `postgresql://user:password@localhost/dbname`

### Clustering Algorithms

- **HDBSCAN** (recommended): Finds clusters of varying sizes
- **K-Means**: Traditional clustering with fixed K

### Embedding Models

- **OpenAI**: High quality, requires API key
- **Sentence Transformers**: Local, free alternative

## Monitoring and Metrics

The system provides several monitoring endpoints:

- `/health`: Health check
- `/stats/overview`: System statistics
- `/clustering/stats`: Clustering metrics

## Troubleshooting

### Common Issues

1. **No embeddings generated**: Check OpenAI API key or sentence-transformers installation
2. **Clustering fails**: Ensure sufficient data points (>= min_cluster_size)
3. **No summaries**: Verify OpenAI API key and GPT-4 access
4. **Database errors**: Check DATABASE_URL and permissions

### Debug Mode

Enable debug logging:

```env
LOG_LEVEL=DEBUG
```

### Sample Data

Use `--sample` flag to generate test data for development and testing.

## API Documentation

When running the API server, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project is provided as-is for demonstration purposes. 