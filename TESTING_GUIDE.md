# 🧪 Copilot Failure Analysis System - Testing Guide

## ✅ **Quick Start Testing (Working Now!)**

### 1. **Basic Functionality Tests**
```bash
# Install minimal dependencies
pip install sqlalchemy python-dotenv

# Run core functionality tests
python test_basic.py
```
**Status: ✅ PASSING** - Core ingestion, error parsing, and fingerprinting work!

### 2. **API Endpoint Tests** 
```bash
# Test API logic
python test_api_simple.py
```
**Status: ✅ PASSING** - API ingestion and stats endpoints work!

---

## 🚀 **Complete System Testing**

### **Option A: With FastAPI Only (Recommended for Initial Testing)**

```bash
# Install API dependencies
pip install fastapi uvicorn pydantic

# Start simplified API server
python test_api_simple.py --server
```

Then test with curl:
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Ingest an event
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
    "ai_output": "I cannot create calendar events."
  }'

# Check stats
curl -X GET "http://localhost:8000/stats/simple"
```

**Interactive API Docs**: Visit `http://localhost:8000/docs`

---

### **Option B: Full System with ML (Advanced)**

```bash
# Install full dependencies (requires C++ build tools)
pip install scikit-learn sentence-transformers hdbscan numpy pandas

# Run complete system with sample data
python main.py --sample

# Start full API
python api.py
```

**Full API Endpoints** (`http://localhost:8000`):
- `GET /health` - Health check
- `POST /ingest/event` - Ingest single event
- `POST /ingest/batch` - Batch ingestion
- `POST /pipeline/process` - Full ML pipeline
- `GET /clusters/top` - **Main endpoint** - Clustered results
- `GET /stats/overview` - System statistics

---

## 📊 **Test Results Summary**

| Component | Status | Notes |
|-----------|--------|-------|
| ✅ Database Models | PASSED | SQLite tables created successfully |
| ✅ Error Parsing | PASSED | Extracts plugin, endpoint, status codes |
| ✅ Fingerprinting | PASSED | Generates consistent fingerprints |
| ✅ Data Ingestion | PASSED | Events stored with metadata |
| ✅ API Endpoints | PASSED | Ingestion and stats working |
| ⏳ ML Components | REQUIRES DEPS | Need scikit-learn, sentence-transformers |
| ⏳ Clustering | REQUIRES DEPS | Need HDBSCAN for clustering |
| ⏳ GPT Summaries | REQUIRES API KEY | Need OpenAI API key |

---

## 🔍 **Manual Testing Scenarios**

### **Scenario 1: Calendar Permission Failures**
```json
{
  "evaluation_id": "cal_001",
  "customer_prompt": "Schedule a team meeting",
  "skill_input": "Create calendar event",
  "skill_output": {
    "error": "GraphAPI.Forbidden",
    "status_code": 403,
    "message": "Insufficient privileges",
    "plugin": "Calendar", 
    "endpoint": "/me/events"
  }
}
```

### **Scenario 2: Email Access Issues**
```json
{
  "evaluation_id": "email_001", 
  "customer_prompt": "Find emails from John",
  "skill_input": "Search emails",
  "skill_output": {
    "error": "GraphAPI.NotFound",
    "status_code": 404,
    "message": "User not found",
    "plugin": "Outlook",
    "endpoint": "/me/messages"
  }
}
```

### **Scenario 3: Network Timeouts**
```json
{
  "evaluation_id": "timeout_001",
  "customer_prompt": "Get my calendar",
  "skill_input": "Retrieve calendar", 
  "skill_output": {
    "error": "NetworkTimeout",
    "status_code": 408,
    "message": "Request timeout",
    "plugin": "Calendar",
    "endpoint": "/me/calendar/events"
  }
}
```

---

## 🛠 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Import Errors**
```
❌ ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: `pip install fastapi uvicorn pydantic`

#### **2. Build Tool Errors** 
```
❌ Microsoft Visual C++ 14.0 or greater is required
```
**Solution**: Use Option A (FastAPI only) or install Visual Studio Build Tools

#### **3. Database Issues**
```
❌ sqlalchemy.exc.OperationalError
```
**Solution**: Check file permissions, SQLite file location

#### **4. API Connection Issues**
```
❌ Connection refused to localhost:8000
```
**Solution**: Ensure server is running with `python test_api_simple.py --server`

---

## 🎯 **Testing Workflows**

### **Development Testing**
1. Run `python test_basic.py` - Verify core functionality
2. Run `python test_api_simple.py` - Test API logic
3. Start API with `python test_api_simple.py --server`
4. Test endpoints with curl or Postman
5. Check `http://localhost:8000/docs` for interactive testing

### **Production Testing**
1. Install full dependencies
2. Set environment variables (OpenAI API key)
3. Run `python main.py --sample --openai`
4. Start full API with `python api.py`
5. Test complete pipeline with `/pipeline/process`
6. Verify results with `/clusters/top`

### **Load Testing**
```bash
# Batch ingest multiple events
curl -X POST "http://localhost:8000/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{"events": [...]}'  # Array of events
```

---

## 📈 **Expected Results**

### **After Basic Testing**
- Database contains sample events
- Error metadata extracted correctly
- Fingerprints generated consistently
- API endpoints respond correctly

### **After Full System Testing**
- Events clustered by failure patterns
- AI-generated summaries available
- REST API returns cluster analysis
- Common patterns identified (plugins, endpoints, error codes)

### **Sample Cluster Output**
```json
{
  "cluster_id": 1,
  "size": 15,
  "algorithm": "hdbscan", 
  "summary": "Calendar permission failures affecting meeting creation",
  "root_cause": "Users lack Graph API permissions for calendar operations",
  "common_plugins": ["Calendar"],
  "common_endpoints": ["/me/events", "/me/onlineMeetings"],
  "common_error_codes": [403],
  "sample_prompts": ["Create a meeting...", "Schedule time..."]
}
```

---

## 🏁 **Success Criteria**

✅ **Core System Working**: Basic tests pass, data ingestion works  
✅ **API Functional**: Endpoints respond, interactive docs available  
⏳ **ML Pipeline**: Clustering and embeddings work (needs dependencies)  
⏳ **AI Summaries**: GPT-4 analysis available (needs API key)  

**Current Status: Core system is fully functional and ready for basic use!** 