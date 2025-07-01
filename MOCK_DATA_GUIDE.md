# Mock Data and Testing Guide

## 📊 Available Mock Data

Yes! The system has comprehensive mock data to test various failure scenarios. Here's what's available:

### 🗂️ Generated Files

| File | Description | Size | Use Case |
|------|-------------|------|----------|
| `sample_telemetry_data.json` | Small sample dataset | 31 events | Quick testing |
| `mock_telemetry_data.json` | Medium dataset with clusters | 100 events | Full testing |
| `demo_large_dataset.json` | Large comprehensive dataset | 200 events | Performance testing |
| `demo_clustered_dataset.json` | Clustered patterns | 80 events | Clustering analysis |
| `sample_telemetry_template.xlsx` | Excel template | 5 samples | Your data format |
| `copilot_failures_template.xlsx` | Excel template | 5 samples | Your data format |

### 🎯 Mock Data Features

The generated data includes:

**Realistic Failure Scenarios:**
- Calendar permission issues (403 Forbidden)
- Outlook resource not found (404)
- Teams authentication failures (401 Unauthorized)
- Network timeouts (408)
- SharePoint access denials
- PowerBI service errors

**Natural Clustering Patterns:**
- Calendar/GraphAPI.Forbidden: 12+ similar events
- Outlook/GraphAPI.NotFound: 8+ similar events  
- Teams/GraphAPI.Unauthorized: 6+ similar events

**Comprehensive Fields:**
```json
{
  "evaluation_id": "mock_0001",
  "customer_prompt": "Create a meeting for next Monday with the team",
  "skill_input": "Execute calendar operation", 
  "skill_output": {
    "error": "GraphAPI.Forbidden",
    "status_code": 403,
    "message": "Error in Calendar: GraphAPI.Forbidden",
    "plugin": "Calendar",
    "endpoint": "/me/events",
    "timestamp": "2025-06-22T09:15:17.574386"
  },
  "ai_output": "I'm sorry, I encountered an error with Calendar."
}
```

## 📝 Using Your Excel Data

### Option 1: Use the Template

1. **Get the template:**
   ```bash
   python excel_import.py create_template
   ```

2. **Fill in your data** using these columns:
   - `evaluation_id`: Unique ID for each failure
   - `customer_prompt`: What the user asked Copilot
   - `skill_input`: How Copilot interpreted it
   - `skill_output`: Error details (JSON or text)
   - `ai_output`: Natural language response (optional)

3. **Import your Excel:**
   ```bash
   python excel_import.py import your_failures.xlsx
   ```

### Option 2: Column Auto-Detection

The Excel importer automatically detects common column variations:
- `id`, `eval_id`, `event_id` → `evaluation_id`
- `prompt`, `user_prompt` → `customer_prompt`
- `input` → `skill_input`
- `output`, `error_output` → `skill_output`
- `response`, `ai_response` → `ai_output`

### Option 3: Flexible Error Formats

Your `skill_output` can be:

**JSON format:**
```json
{"error": "GraphAPI.Forbidden", "status_code": 403, "plugin": "Calendar"}
```

**Plain text:**
```
Calendar plugin error: 403 Forbidden at /me/events
Teams timeout error - request failed after 30 seconds
```

The system will automatically parse and extract:
- Error types (Forbidden, NotFound, Unauthorized, etc.)
- Status codes (400, 401, 403, 404, 500, etc.)
- Plugin names (Calendar, Outlook, Teams, etc.)
- Common patterns for clustering

## 🧪 Testing the System

### Quick Start
```bash
# Generate mock data and test
python demo.py

# Test basic functionality
python test_basic.py

# Test API endpoints
python test_api_simple.py
```

### Manual Testing
```bash
# 1. Generate mock data
python mock_data.py

# 2. Create Excel template
python excel_import.py create_template

# 3. Start API server
python api.py

# 4. Test endpoints
curl http://localhost:8000/stats/overview
```

## 📈 Analysis Results

With the mock data, you can immediately see:

**Failure Patterns:**
```
Calendar/GraphAPI.Forbidden: 12 occurrences
Outlook/GraphAPI.NotFound: 8 occurrences  
Teams/GraphAPI.Unauthorized: 6 occurrences
```

**Common Endpoints:**
- `/me/events` (Calendar failures)
- `/me/messages` (Outlook issues)
- `/me/teams` (Teams problems)

**Error Types Distribution:**
- 40% Permission issues (403)
- 25% Resource not found (404)
- 20% Authentication (401)
- 15% Timeouts (408)

## 🔧 Customizing Mock Data

Edit `mock_data.py` to:
- Add your specific plugins
- Include your error patterns
- Match your endpoint structures
- Use your prompt formats

Example customization:
```python
# Add your plugins
plugins = ["Calendar", "Outlook", "Teams", "YourCustomPlugin"]

# Add your error types  
error_types = ["GraphAPI.Forbidden", "YourCustomError"]

# Add your endpoints
endpoints = {
    "YourCustomPlugin": ["/api/custom", "/api/other"]
}
```

## 🚀 Next Steps

1. **Review the generated files** - Look at the JSON examples
2. **Try the Excel template** - Fill it with a few test rows
3. **Run the demo** - See the full pipeline in action
4. **Start with your data** - Use the Excel import when ready
5. **Scale up** - The system handles thousands of events efficiently

The mock data gives you everything needed to test clustering, analyze patterns, and verify the system works before adding your real telemetry data! 