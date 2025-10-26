# AdventureWorks Analytics API - Deployment Guide
**Phase 7: Real-Time API & Integration**

**Version:** 1.0
**Date:** October 24, 2025
**Status:** Production Ready

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Local Development](#local-development)
4. [API Documentation](#api-documentation)
5. [Testing](#testing)
6. [Production Deployment](#production-deployment)
7. [CRM Integration](#crm-integration)
8. [Security](#security)
9. [Monitoring](#monitoring)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Dependencies
```bash
# Activate virtual environment
source venv/bin/activate

# Install Phase 7 packages
pip install fastapi uvicorn[standard] pydantic requests jinja2 python-multipart aiofiles
```

### 2. Start API Server
```bash
# Development mode (auto-reload)
uvicorn api.main:app --reload --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Access API
- **API Root:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/api/docs (Swagger UI)
- **Alternative Docs:** http://localhost:8000/api/redoc (ReDoc)

### 4. Test API
```bash
# Run comprehensive test suite
python api/test_api.py
```

---

## Installation

### System Requirements
- **Python:** 3.9+ (tested on 3.13)
- **RAM:** 2GB minimum, 4GB recommended
- **Disk:** 500MB for API + models
- **OS:** macOS, Linux, Windows

### Phase 7 Dependencies
```txt
fastapi                 # Modern API framework
uvicorn[standard]       # ASGI server
pydantic               # Data validation
requests               # HTTP client
jinja2                 # Email templates
python-multipart       # Form data
aiofiles               # Async file operations
```

### Installation Steps

**Option 1: Update existing environment**
```bash
pip install -r requirements.txt
```

**Option 2: Install manually**
```bash
pip install fastapi uvicorn[standard] pydantic requests jinja2 python-multipart aiofiles
```

**Verify installation:**
```bash
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
python -c "import uvicorn; print('Uvicorn installed')"
```

---

## Local Development

### Start Development Server
```bash
cd /Users/gaganjit/Documents/AdventureWorks

# Activate virtual environment
source venv/bin/activate

# Start with auto-reload
uvicorn api.main:app --reload --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Test Endpoints

**Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

**Churn Prediction (with API key):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/churn" \
  -H "X-API-Key: dev_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 12345,
    "recency": 90,
    "frequency": 10,
    "monetary": 5000
  }'
```

**Response:**
```json
{
  "customer_id": 12345,
  "churn_probability": 0.7543,
  "risk_level": "High",
  "recommendation": "Customer inactive 90 days. Launch win-back campaign with 15% discount.",
  "timestamp": "2025-10-24T10:30:00",
  "model_used": "Random Forest (Optimized)"
}
```

---

## API Documentation

### Endpoints Overview

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/` | GET | Root endpoint | No |
| `/api/v1/health` | GET | Health check | No |
| `/api/v1/models/status` | GET | Model metadata | Yes |
| `/api/v1/predict/churn` | POST | Customer churn prediction | Yes |
| `/api/v1/predict/return-risk` | POST | Product return risk | Yes |
| `/api/v1/customers/{id}/risk` | GET | Get customer risk score | Yes |
| `/api/v1/predict/batch` | POST | Batch predictions (up to 1000) | Yes |
| `/api/v1/alerts/subscribe` | POST | Subscribe to real-time alerts | Yes |

### Interactive Documentation

**Swagger UI (Recommended):**
```
http://localhost:8000/api/docs
```

Features:
- ✅ Try endpoints directly in browser
- ✅ View request/response schemas
- ✅ Automatic validation
- ✅ Download OpenAPI spec

**ReDoc (Alternative):**
```
http://localhost:8000/api/redoc
```

Features:
- ✅ Clean, readable documentation
- ✅ Search functionality
- ✅ Code samples

### Authentication

**API Key Header:**
```http
X-API-Key: your-api-key-here
```

**Available Keys (Development):**
- `dev_key_12345` - Basic tier (development)
- `prod_key_67890` - Premium tier (production)

**Generate New Keys:**
```python
import secrets
new_key = secrets.token_urlsafe(32)
print(new_key)
```

**Production:** Store keys in environment variables or database.

---

## Testing

### Automated Test Suite

**Run all tests:**
```bash
python api/test_api.py
```

**Tests included:**
1. ✅ Root endpoint
2. ✅ Health check
3. ✅ Models status
4. ✅ Churn prediction (low risk)
5. ✅ Churn prediction (high risk)
6. ✅ Return risk (Bikes)
7. ✅ Return risk (Accessories)
8. ✅ Batch predictions
9. ✅ Alert subscription
10. ✅ Authentication (invalid key)
11. ✅ Input validation
12. ✅ Performance (<1s response time)

**Expected output:**
```
====================================================================
TEST SUMMARY
====================================================================

PASS  test_1_root_endpoint
PASS  test_2_health_check
PASS  test_3_models_status
...
Results: 12/12 tests passed (100%)

✅ ALL TESTS PASSED!
```

### Manual Testing

**Python Client:**
```bash
python api/examples/python_client.py
```

Runs 7 examples:
- Single churn prediction
- Return risk prediction
- Batch predictions
- CRM integration
- Alert subscription
- Model monitoring
- Webhook integration

### Performance Testing

**Apache Bench:**
```bash
ab -n 1000 -c 10 \
  -H "X-API-Key: dev_key_12345" \
  -p payload.json \
  -T application/json \
  http://localhost:8000/api/v1/predict/churn
```

**Locust (Load Testing):**
```python
# locustfile.py
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)
    headers = {"X-API-Key": "dev_key_12345"}

    @task
    def predict_churn(self):
        self.client.post("/api/v1/predict/churn",
            json={"recency": 90, "frequency": 10, "monetary": 5000},
            headers=self.headers)
```

Run: `locust -f locustfile.py --host=http://localhost:8000`

---

## Production Deployment

### Option 1: Docker (Recommended)

**Create Dockerfile:**
```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with gunicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Build and run:**
```bash
# Build image
docker build -t adventureworks-api .

# Run container
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name adventureworks-api \
  adventureworks-api

# Check logs
docker logs -f adventureworks-api
```

**Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - API_ENV=production
      - LOG_LEVEL=info
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
```

Run: `docker-compose up -d`

---

### Option 2: Cloud Platform (AWS, Azure, GCP)

**AWS Elastic Beanstalk:**
```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.13 adventureworks-api

# Create environment
eb create adventureworks-api-prod

# Deploy
eb deploy
```

**Azure App Service:**
```bash
# Login
az login

# Create resource group
az group create --name adventureworks-rg --location eastus

# Create app service
az webapp up --name adventureworks-api --resource-group adventureworks-rg --runtime "PYTHON:3.13"
```

**Google Cloud Run:**
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/adventureworks-api

# Deploy
gcloud run deploy adventureworks-api \
  --image gcr.io/PROJECT_ID/adventureworks-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

### Option 3: Linux Server (VPS/VM)

**1. Install system packages:**
```bash
sudo apt update
sudo apt install python3.13 python3-pip nginx supervisor
```

**2. Setup application:**
```bash
cd /var/www/adventureworks
git clone <your-repo>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Configure Supervisor:**
```ini
# /etc/supervisor/conf.d/adventureworks-api.conf
[program:adventureworks-api]
command=/var/www/adventureworks/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 4
directory=/var/www/adventureworks
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/adventureworks/api.err.log
stdout_logfile=/var/log/adventureworks/api.out.log
```

**4. Configure Nginx:**
```nginx
# /etc/nginx/sites-available/adventureworks-api
server {
    listen 80;
    server_name api.adventureworks.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

**5. Enable and start:**
```bash
sudo ln -s /etc/nginx/sites-available/adventureworks-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start adventureworks-api
```

**6. Setup SSL (Let's Encrypt):**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.adventureworks.com
```

---

## CRM Integration

### Salesforce

**1. Create custom fields:**
- `Churn_Score__c` (Number)
- `Churn_Risk_Level__c` (Picklist: Low, Medium, High)
- `Last_Churn_Update__c` (DateTime)

**2. Setup webhook:**
```python
from api.examples.crm_webhooks import SalesforceIntegration

sf = SalesforceIntegration(
    instance_url="https://yourinstance.salesforce.com",
    access_token="your-oauth-token"
)

sf.update_customer_churn_score(
    customer_email="john.doe@example.com",
    churn_probability=0.85,
    risk_level="High"
)
```

**3. Automate with Process Builder:**
- Trigger: Contact updated
- Criteria: Churn_Score__c >= 70
- Action: Create Task for account owner

---

### HubSpot

**1. Create custom properties:**
```bash
curl -X POST https://api.hubapi.com/crm/v3/properties/contacts \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "churn_score",
    "label": "Churn Score",
    "type": "number",
    "fieldType": "number"
  }'
```

**2. Update contacts:**
```python
from api.examples.crm_webhooks import HubSpotIntegration

hs = HubSpotIntegration(api_key="your-hubspot-api-key")

hs.update_contact_churn_score(
    email="customer@example.com",
    churn_probability=0.75,
    risk_level="High"
)
```

**3. Create workflow:**
- Enrollment trigger: `churn_score >= 70`
- Action: Send internal email notification
- Action: Create task for sales rep

---

### Generic Webhook

**Setup webhook receiver:**
```python
# webhook_receiver.py (Flask example)
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhooks/churn', methods=['POST'])
def receive_churn():
    data = request.json
    customer_id = data['customer_id']
    risk_level = data['risk_level']

    # Your custom logic here
    print(f"High-risk customer: {customer_id} - {risk_level}")

    return jsonify({"status": "received"}), 200

if __name__ == '__main__':
    app.run(port=5000)
```

**Configure API to send webhooks:**
```python
from api.notifications import send_churn_webhook

send_churn_webhook(
    webhook_url="https://your-system.com/webhooks/churn",
    churn_data={
        "customer_id": 12345,
        "churn_probability": 0.85,
        "risk_level": "High"
    }
)
```

---

## Security

### API Key Management

**Production setup:**
```python
# Use environment variables
import os

VALID_API_KEYS = {
    os.getenv("API_KEY_PROD"): {"name": "Production", "tier": "premium"},
    os.getenv("API_KEY_PARTNER"): {"name": "Partner", "tier": "basic"},
}
```

**Rotate keys regularly:**
```bash
# Generate new key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update in database/env vars
# Notify clients
# Deactivate old key after grace period
```

### HTTPS/TLS

**Let's Encrypt (Free):**
```bash
sudo certbot --nginx -d api.adventureworks.com
```

**Or use Cloud Platform SSL:**
- AWS: Application Load Balancer + ACM
- Azure: App Service managed certificate
- GCP: Cloud Run auto-provisions SSL

### Rate Limiting

**Add to main.py:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/v1/predict/churn")
@limiter.limit("100/hour")
async def predict_churn(...):
    ...
```

Install: `pip install slowapi`

### CORS Configuration

**Update main.py:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dashboard.adventureworks.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## Monitoring

### Health Checks

**Uptime monitoring:**
```bash
# Cron job (every 5 minutes)
*/5 * * * * curl -f http://localhost:8000/api/v1/health || echo "API down!" | mail -s "Alert" ops@adventureworks.com
```

**UptimeRobot/Pingdom:**
- URL: `https://api.adventureworks.com/api/v1/health`
- Interval: 5 minutes
- Alert: Email/SMS on failure

### Logging

**Configure logging in main.py:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/adventureworks/api.log'),
        logging.StreamHandler()
    ]
)
```

**View logs:**
```bash
# Docker
docker logs -f adventureworks-api

# Supervisor
tail -f /var/log/adventureworks/api.out.log

# Systemd
journalctl -u adventureworks-api -f
```

### Metrics

**Prometheus + Grafana:**
```bash
pip install prometheus-fastapi-instrumentator

# In main.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Access metrics: `http://localhost:8000/metrics`

---

## Troubleshooting

### Issue: API won't start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Error:** `Address already in use`
```bash
# Solution: Find and kill process
lsof -i :8000
kill -9 <PID>
```

---

### Issue: Models not loading

**Error:** `FileNotFoundError: models/churn_prediction/random_forest.pkl`
```bash
# Solution: Train models first
python scripts/phase3_models_training.py
python scripts/phase4_models_training.py
python scripts/phase5_optimization.py
```

---

### Issue: Predictions failing

**Check model paths:**
```python
import joblib
from pathlib import Path

model_path = Path("models/churn_prediction/random_forest_optimized.pkl")
print(f"Exists: {model_path.exists()}")

if model_path.exists():
    model = joblib.load(model_path)
    print(f"Loaded: {type(model)}")
```

---

### Issue: Slow response times

**Check:**
1. Model caching enabled? (should be in main.py)
2. Too many workers? (reduce if low RAM)
3. Database queries slow? (optimize data loading)

**Optimize:**
```python
# Pre-load models on startup
@app.on_event("startup")
async def load_models():
    load_model(churn_model_path)
    load_model(return_model_path)
```

---

## Next Steps

1. ✅ **Test locally:** Run `python api/test_api.py`
2. ⏸️ **Deploy to cloud:** Choose AWS/Azure/GCP
3. ⏸️ **Integrate with CRM:** Setup Salesforce/HubSpot webhooks
4. ⏸️ **Enable monitoring:** Setup Prometheus + Grafana
5. ⏸️ **Configure alerts:** Email/SMS for API downtime
6. ⏸️ **Load testing:** Test with 1000+ concurrent users

---

## Support

**Documentation:**
- Swagger UI: http://localhost:8000/api/docs
- Python Examples: `api/examples/python_client.py`
- CRM Examples: `api/examples/crm_webhooks.py`

**Contact:**
- Email: api-support@adventureworks.com
- Slack: #analytics-api
- Issues: https://github.com/adventureworks/analytics/issues

---

**Last Updated:** October 24, 2025
**API Version:** 1.0.0
**Status:** ✅ Production Ready
