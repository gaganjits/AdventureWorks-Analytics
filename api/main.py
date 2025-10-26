"""
AdventureWorks Analytics API
Phase 7: Real-Time API & Integration

FastAPI application providing REST endpoints for:
- Revenue forecasting
- Customer churn prediction
- Product return risk assessment
- Real-time alerts and notifications

Author: AdventureWorks Data Science Team
Version: 1.0
Date: October 24, 2025
"""

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import hashlib
import secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AdventureWorks Analytics API",
    description="Real-time predictions and analytics for revenue, churn, and return risk",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# API Keys (In production, store in environment variables or database)
VALID_API_KEYS = {
    "dev_key_12345": {"name": "Development", "tier": "basic"},
    "prod_key_67890": {"name": "Production", "tier": "premium"},
    # Generate new keys with: secrets.token_urlsafe(32)
}

# Model paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load models (lazy loading)
MODELS_CACHE = {}

def load_model(model_path: Path):
    """Load model with caching"""
    if str(model_path) not in MODELS_CACHE:
        logger.info(f"Loading model: {model_path}")
        MODELS_CACHE[str(model_path)] = joblib.load(model_path)
    return MODELS_CACHE[str(model_path)]

# ============================================================================
# AUTHENTICATION
# ============================================================================

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> Dict[str, Any]:
    """Verify API key"""
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key required. Include 'X-API-Key' header."
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )

    return VALID_API_KEYS[api_key]

# ============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    models_loaded: int

class ModelStatusResponse(BaseModel):
    model_name: str
    model_type: str
    accuracy: Optional[float]
    last_trained: Optional[str]
    features_count: int
    status: str

class RevenueRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    features: Optional[Dict[str, float]] = Field(None, description="Optional engineered features")

    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

class RevenueResponse(BaseModel):
    prediction: float
    confidence_interval: Dict[str, float]
    timestamp: datetime
    model_used: str

class ChurnRequest(BaseModel):
    customer_id: Optional[int] = None
    recency: float = Field(..., ge=0, description="Days since last purchase")
    frequency: int = Field(..., ge=0, description="Total number of purchases")
    monetary: float = Field(..., ge=0, description="Total revenue from customer")
    avg_order_value: Optional[float] = Field(None, ge=0)
    total_quantity: Optional[int] = Field(None, ge=0)
    annual_income: Optional[float] = None
    total_children: Optional[int] = None
    education_level: Optional[str] = None
    occupation: Optional[str] = None
    home_owner: Optional[str] = None

class ChurnResponse(BaseModel):
    customer_id: Optional[int]
    churn_probability: float
    risk_level: str  # "Low", "Medium", "High"
    recommendation: str
    timestamp: datetime
    model_used: str

class ReturnRiskRequest(BaseModel):
    product_id: Optional[int] = None
    product_name: Optional[str] = None
    category: str = Field(..., description="Product category: Bikes, Clothing, Accessories")
    subcategory: str = Field(..., description="Product subcategory")
    list_price: float = Field(..., ge=0)
    return_rate: Optional[float] = Field(None, ge=0, le=1)
    sales_volume: Optional[int] = Field(None, ge=0)

    @validator('category')
    def validate_category(cls, v):
        valid = ["Bikes", "Clothing", "Accessories"]
        if v not in valid:
            raise ValueError(f'Category must be one of {valid}')
        return v

class ReturnRiskResponse(BaseModel):
    product_id: Optional[int]
    product_name: Optional[str]
    risk_probability: float
    risk_level: str  # "Low", "High"
    recommendation: str
    timestamp: datetime
    model_used: str

class BatchPredictionRequest(BaseModel):
    predictions: List[Dict[str, Any]]
    prediction_type: str  # "churn", "return_risk", "revenue"

class BatchPredictionResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_processed: int
    timestamp: datetime

class AlertSubscription(BaseModel):
    email: str
    alert_types: List[str]  # ["churn", "return_risk", "revenue_drop"]
    threshold: Optional[float] = 0.7
    webhook_url: Optional[str] = None

class AlertResponse(BaseModel):
    subscription_id: str
    status: str
    message: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"

def get_churn_recommendation(risk_level: str, recency: float, monetary: float) -> str:
    """Generate actionable recommendation based on churn risk"""
    if risk_level == "High":
        if monetary > 1000:
            return "High-value customer at risk! Personal outreach recommended within 24 hours."
        elif recency > 120:
            return "Customer inactive 120+ days. Launch win-back campaign with 15% discount."
        else:
            return "Send retention offer: 10% discount on next purchase within 7 days."
    elif risk_level == "Medium":
        return "Monitor customer. Consider engagement campaign (newsletter, product recommendations)."
    else:
        return "Customer healthy. Continue standard engagement."

def get_return_risk_recommendation(risk_level: str, category: str) -> str:
    """Generate actionable recommendation based on return risk"""
    if risk_level == "High":
        if category == "Bikes":
            return "High return risk detected. Schedule quality audit and review customer feedback."
        elif category == "Clothing":
            return "Potential sizing issue. Review size chart accuracy and customer reviews."
        else:
            return "Monitor product closely. Consider quality control inspection."
    else:
        return "Product performing well. Continue standard monitoring."

def prepare_churn_features(request: ChurnRequest) -> pd.DataFrame:
    """Prepare features for churn prediction"""
    features = {
        'Recency': request.recency,
        'Frequency': request.frequency,
        'Monetary': request.monetary,
        'AvgOrderValue': request.avg_order_value or (request.monetary / max(request.frequency, 1)),
        'TotalQuantity': request.total_quantity or request.frequency * 2,  # Estimate
        'AnnualIncome': request.annual_income or 60000,  # Default median
        'TotalChildren': request.total_children or 0,
    }

    # Create DataFrame
    df = pd.DataFrame([features])

    # Load the actual feature list from training
    try:
        feature_names_path = MODELS_DIR / "churn_prediction" / "feature_names.pkl"
        if feature_names_path.exists():
            expected_features = joblib.load(feature_names_path)
            # Add missing features with default values
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            # Reorder to match training
            df = df[expected_features]
    except Exception as e:
        logger.warning(f"Could not load feature names: {e}")

    return df

def prepare_return_risk_features(request: ReturnRiskRequest) -> pd.DataFrame:
    """Prepare features for return risk prediction"""
    # Map category to binary features
    category_map = {
        'Bikes': {'Category_Bikes': 1, 'Category_Clothing': 0, 'Category_Accessories': 0},
        'Clothing': {'Category_Bikes': 0, 'Category_Clothing': 1, 'Category_Accessories': 0},
        'Accessories': {'Category_Bikes': 0, 'Category_Clothing': 0, 'Category_Accessories': 1},
    }

    features = {
        'ListPrice': request.list_price,
        'ReturnRate': request.return_rate or 0.02,  # Default to average
        'SalesVolume': request.sales_volume or 50,  # Default estimate
        **category_map.get(request.category, category_map['Accessories'])
    }

    df = pd.DataFrame([features])

    # Load expected features
    try:
        feature_names_path = MODELS_DIR / "return_risk" / "feature_names.pkl"
        if feature_names_path.exists():
            expected_features = joblib.load(feature_names_path)
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[expected_features]
    except Exception as e:
        logger.warning(f"Could not load feature names: {e}")

    return df

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AdventureWorks Analytics API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "status": "operational"
    }

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded=len(MODELS_CACHE)
    )

@app.get("/api/v1/models/status", response_model=List[ModelStatusResponse], tags=["Models"])
async def models_status(api_key: Dict = Depends(verify_api_key)):
    """Get status of all available models"""
    models_info = []

    # Revenue model
    revenue_model_path = MODELS_DIR / "revenue_forecasting" / "xgboost_optimized.pkl"
    if revenue_model_path.exists():
        models_info.append(ModelStatusResponse(
            model_name="Revenue Forecasting",
            model_type="XGBoost (Optimized)",
            accuracy=11.58,  # MAPE
            last_trained="2025-10-24",
            features_count=32,
            status="active"
        ))

    # Churn model
    churn_model_path = MODELS_DIR / "churn_prediction" / "random_forest_optimized.pkl"
    if churn_model_path.exists():
        models_info.append(ModelStatusResponse(
            model_name="Churn Prediction",
            model_type="Random Forest (Optimized)",
            accuracy=100.0,  # Accuracy
            last_trained="2025-10-24",
            features_count=15,
            status="active"
        ))

    # Return risk model
    return_model_path = MODELS_DIR / "return_risk" / "xgboost_optimized.pkl"
    if return_model_path.exists():
        models_info.append(ModelStatusResponse(
            model_name="Return Risk",
            model_type="XGBoost (Optimized)",
            accuracy=100.0,  # Accuracy
            last_trained="2025-10-24",
            features_count=29,
            status="active"
        ))

    return models_info

@app.post("/api/v1/predict/churn", response_model=ChurnResponse, tags=["Predictions"])
async def predict_churn(
    request: ChurnRequest,
    api_key: Dict = Depends(verify_api_key)
):
    """
    Predict customer churn probability

    Returns churn probability (0-1), risk level, and actionable recommendation.
    """
    try:
        # Load model
        model_path = MODELS_DIR / "churn_prediction" / "random_forest_optimized.pkl"
        if not model_path.exists():
            model_path = MODELS_DIR / "churn_prediction" / "random_forest.pkl"

        model = load_model(model_path)

        # Prepare features
        features_df = prepare_churn_features(request)

        # Make prediction
        churn_prob = model.predict_proba(features_df)[0][1]  # Probability of churn (class 1)

        # Calculate risk level
        risk_level = calculate_risk_level(churn_prob)

        # Get recommendation
        recommendation = get_churn_recommendation(risk_level, request.recency, request.monetary)

        logger.info(f"Churn prediction: customer_id={request.customer_id}, prob={churn_prob:.3f}, risk={risk_level}")

        return ChurnResponse(
            customer_id=request.customer_id,
            churn_probability=round(churn_prob, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            timestamp=datetime.now(),
            model_used="Random Forest (Optimized)"
        )

    except Exception as e:
        logger.error(f"Churn prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/v1/predict/return-risk", response_model=ReturnRiskResponse, tags=["Predictions"])
async def predict_return_risk(
    request: ReturnRiskRequest,
    api_key: Dict = Depends(verify_api_key)
):
    """
    Predict product return risk

    Returns return probability (0-1), risk level, and actionable recommendation.
    """
    try:
        # Load model
        model_path = MODELS_DIR / "return_risk" / "xgboost_optimized.pkl"
        if not model_path.exists():
            model_path = MODELS_DIR / "return_risk" / "xgboost.pkl"

        model = load_model(model_path)

        # Prepare features
        features_df = prepare_return_risk_features(request)

        # Make prediction
        return_prob = model.predict_proba(features_df)[0][1]  # Probability of high return risk

        # Risk level (binary for return risk)
        risk_level = "High" if return_prob >= 0.5 else "Low"

        # Get recommendation
        recommendation = get_return_risk_recommendation(risk_level, request.category)

        logger.info(f"Return risk prediction: product_id={request.product_id}, prob={return_prob:.3f}, risk={risk_level}")

        return ReturnRiskResponse(
            product_id=request.product_id,
            product_name=request.product_name,
            risk_probability=round(return_prob, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            timestamp=datetime.now(),
            model_used="XGBoost (Optimized)"
        )

    except Exception as e:
        logger.error(f"Return risk prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/v1/customers/{customer_id}/risk", response_model=ChurnResponse, tags=["Customers"])
async def get_customer_risk(
    customer_id: int,
    api_key: Dict = Depends(verify_api_key)
):
    """
    Get churn risk for a specific customer (requires customer data lookup)

    In production, this would query your customer database for RFM values.
    For demo, using sample values.
    """
    # In production: query customer database for actual RFM values
    # For now, return demo prediction

    # Load processed customer data
    try:
        customer_rfm_path = BASE_DIR / "data" / "processed" / "Customer_RFM.csv"
        if customer_rfm_path.exists():
            df = pd.read_csv(customer_rfm_path)
            customer_data = df[df['CustomerKey'] == customer_id]

            if customer_data.empty:
                raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

            # Extract RFM values
            request = ChurnRequest(
                customer_id=customer_id,
                recency=float(customer_data['Recency'].iloc[0]),
                frequency=int(customer_data['Frequency'].iloc[0]),
                monetary=float(customer_data['Monetary'].iloc[0]),
                avg_order_value=float(customer_data.get('AvgOrderValue', customer_data['Monetary'] / customer_data['Frequency']).iloc[0]),
                annual_income=float(customer_data.get('AnnualIncome', 60000).iloc[0]) if 'AnnualIncome' in customer_data.columns else 60000,
            )

            return await predict_churn(request, api_key)
        else:
            raise HTTPException(status_code=503, detail="Customer data not available")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Customer lookup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Customer lookup failed: {str(e)}")

@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predictions(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: Dict = Depends(verify_api_key)
):
    """
    Process batch predictions (up to 1000 records)

    For larger batches, consider using background tasks or async processing.
    """
    if len(request.predictions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limited to 1000 records")

    results = []

    try:
        for item in request.predictions:
            if request.prediction_type == "churn":
                req = ChurnRequest(**item)
                pred = await predict_churn(req, api_key)
                results.append(pred.dict())
            elif request.prediction_type == "return_risk":
                req = ReturnRiskRequest(**item)
                pred = await predict_return_risk(req, api_key)
                results.append(pred.dict())
            else:
                raise HTTPException(status_code=400, detail=f"Invalid prediction_type: {request.prediction_type}")

        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            timestamp=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/api/v1/alerts/subscribe", response_model=AlertResponse, tags=["Alerts"])
async def subscribe_to_alerts(
    subscription: AlertSubscription,
    api_key: Dict = Depends(verify_api_key)
):
    """
    Subscribe to real-time alerts (email or webhook)

    Alert types: "churn", "return_risk", "revenue_drop"
    Threshold: 0-1 (e.g., 0.7 = alert when risk >= 70%)
    """
    # Generate subscription ID
    subscription_id = hashlib.sha256(
        f"{subscription.email}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]

    # In production: store subscription in database
    logger.info(f"Alert subscription created: {subscription_id} for {subscription.email}")

    # TODO: Implement actual alert logic (email/webhook)

    return AlertResponse(
        subscription_id=subscription_id,
        status="active",
        message=f"Subscribed to {len(subscription.alert_types)} alert types"
    )

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("=== AdventureWorks Analytics API Starting ===")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")

    # Pre-load critical models
    try:
        churn_model = MODELS_DIR / "churn_prediction" / "random_forest_optimized.pkl"
        if not churn_model.exists():
            churn_model = MODELS_DIR / "churn_prediction" / "random_forest.pkl"
        if churn_model.exists():
            load_model(churn_model)
            logger.info("✓ Churn model loaded")

        return_model = MODELS_DIR / "return_risk" / "xgboost_optimized.pkl"
        if not return_model.exists():
            return_model = MODELS_DIR / "return_risk" / "xgboost.pkl"
        if return_model.exists():
            load_model(return_model)
            logger.info("✓ Return risk model loaded")

        logger.info(f"=== API Ready - {len(MODELS_CACHE)} models loaded ===")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("=== AdventureWorks Analytics API Shutting Down ===")
    MODELS_CACHE.clear()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
