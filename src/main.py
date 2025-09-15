from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import joblib
import numpy as np
import pandas as pd
import io
import traceback
from datetime import datetime
import uvicorn
from groq import Groq
from dotenv import load_dotenv

# =====================
# Load environment
# =====================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =====================
# FastAPI App Setup
# =====================
app = FastAPI(
    title="FraudLens API",
    description="Advanced Fraud Detection API using LightGBM with LLM explanations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Model Configuration
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATHS = {
    'final': os.path.join(BASE_DIR, "fraud_artifacts-20250912T170121Z-1-001", "fraud_artifacts", "final_fraud_model.pkl"),
    'preprocessing': os.path.join(BASE_DIR, "fraud_artifacts-20250912T170121Z-1-001", "fraud_artifacts", "preprocessing.joblib")
}

# Global variables for loaded models
models = {}
preprocessing = {}

# =====================
# Pydantic Models
# =====================
class TransactionData(BaseModel):
    """Transaction data model for fraud detection"""
    customer_age: int = Field(..., ge=0, le=120, description="Customer age in years")
    income: float = Field(..., ge=0, le=1, description="Income level (0-1)")
    name_email_similarity: float = Field(..., ge=0, le=1, description="Name-email similarity")
    prev_address_months_count: int = Field(..., ge=-1, description="Months at previous address (-1 if unknown)")
    current_address_months_count: int = Field(..., ge=0, description="Months at current address")
    days_since_request: float = Field(..., ge=0, description="Days since credit request")
    intended_balcon_amount: float = Field(..., description="Intended balance amount")
    proposed_credit_limit: float = Field(..., ge=0, description="Proposed credit limit")
    payment_type: str = Field(..., description="Payment type (AA, AB, AC, etc.)")
    bank_months_count: int = Field(..., ge=-1, description="Months as bank customer (-1 if unknown)")
    has_other_cards: int = Field(..., ge=0, le=1, description="Has other credit cards (0/1)")
    foreign_request: int = Field(..., ge=0, le=1, description="Foreign transaction request (0/1)")
    velocity_6h: float = Field(..., ge=0, description="Transaction velocity in 6 hours")
    velocity_24h: float = Field(..., ge=0, description="Transaction velocity in 24 hours")
    velocity_4w: float = Field(..., ge=0, description="Transaction velocity in 4 weeks")
    zip_count_4w: int = Field(..., ge=0, description="ZIP code count in last 4 weeks")
    bank_branch_count_8w: int = Field(..., ge=0, description="Bank branch visits in 8 weeks")
    date_of_birth_distinct_emails_4w: int = Field(..., ge=0, description="Distinct emails in 4 weeks")
    credit_risk_score: int = Field(..., ge=0, le=300, description="Credit risk score (0-300)")
    employment_status: str = Field(..., description="Employment status (CA, CB, CC, etc.)")
    housing_status: str = Field(..., description="Housing status (BA, BB, BC, etc.)")
    email_is_free: int = Field(..., ge=0, le=1, description="Free email indicator (0/1)")
    phone_home_valid: int = Field(..., ge=0, le=1, description="Home phone validity (0/1)")
    phone_mobile_valid: int = Field(..., ge=0, le=1, description="Mobile phone validity (0/1)")
    source: str = Field(..., description="Transaction source (INTERNET, MOBILE, BRANCH, ATM)")
    session_length_in_minutes: float = Field(..., ge=0, description="Session duration in minutes")
    device_os: str = Field(..., description="Device OS (windows, mac, linux, android, ios, other)")
    keep_alive_session: int = Field(..., ge=0, le=1, description="Keep alive session (0/1)")
    device_distinct_emails_8w: int = Field(..., ge=0, description="Distinct emails from device in 8 weeks")
    device_fraud_count: int = Field(..., ge=0, description="Fraud count from this device")
    month: int = Field(..., ge=0, le=11, description="Month of transaction (0-11)")
    fraud_bool: Optional[int] = Field(None, ge=0, le=1, description="Ground truth fraud label (optional)")

class FraudPrediction(BaseModel):
    """Fraud prediction response model"""
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    explanation: str = Field(..., description="Human-readable explanation")
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    transactions: List[TransactionData] = Field(..., description="List of transactions to analyze")
    include_explanations: bool = Field(True, description="Include AI-generated explanations")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    predictions: List[FraudPrediction] = Field(..., description="List of fraud predictions")
    model_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    processing_time: float = Field(..., description="Processing time in seconds")
    total_records: int = Field(..., description="Total number of records processed")
    fraud_count: int = Field(..., description="Number of fraud cases detected")

class ExplanationRequest(BaseModel):
    """Request model for generating LLM explanation"""
    transaction_data: TransactionData = Field(..., description="Transaction data for explanation")
    prediction: FraudPrediction = Field(..., description="Existing prediction without explanation")

class ExplanationResponse(BaseModel):
    """Response model for LLM explanation"""
    explanation: str = Field(..., description="Generated LLM explanation")
    processing_time: float = Field(..., description="Time taken to generate explanation")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Current timestamp")

# =====================
# Model Loading Functions
# =====================
def load_models():
    """Load the fraud detection model and preprocessing pipeline"""
    global models, preprocessing
    
    try:
        # Load preprocessing pipeline
        if os.path.exists(MODEL_PATHS['preprocessing']):
            preprocessing = joblib.load(MODEL_PATHS['preprocessing'])
            print("✅ Preprocessing pipeline loaded")
        else:
            raise FileNotFoundError(f"Preprocessing file not found: {MODEL_PATHS['preprocessing']}")
        
        # Load main model
        if os.path.exists(MODEL_PATHS['final']):
            models['final'] = joblib.load(MODEL_PATHS['final'])
            print("✅ Final fraud model loaded")
        else:
            raise FileNotFoundError(f"Model file not found: {MODEL_PATHS['final']}")
        
        print("🎯 All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """Preprocess transaction data for model prediction"""
    try:
        # Get preprocessing components
        encoder = preprocessing["encoder"]
        scaler = preprocessing["scaler"]
        num_cols = preprocessing["num_cols"]
        cat_cols = preprocessing["cat_cols"]
        
        # Make a copy and handle missing values
        df_processed = df.copy()
        df_processed[num_cols] = df_processed[num_cols].fillna(0)
        
        if len(cat_cols) > 0:
            df_processed[cat_cols] = df_processed[cat_cols].astype(str).fillna("##NA##")
        
        # Scale numerical features
        X_num_scaled = scaler.transform(df_processed[num_cols]) if len(num_cols) > 0 else np.zeros((len(df_processed), 0))
        
        # Encode categorical features
        X_cat_enc = encoder.transform(df_processed[cat_cols]) if len(cat_cols) > 0 else np.zeros((len(df_processed), 0))
        
        # Combine features
        X_combined = np.hstack([X_num_scaled, X_cat_enc]).astype(np.float32)
        
        return X_combined
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def generate_llm_justification(features: List[Dict[str, Any]], is_fraud: bool, probability: float, transaction_id: str = None) -> str:
    """Generate natural language explanation using Groq LLM"""
    try:
        if not GROQ_API_KEY:
            return "LLM explanation not available (missing GROQ_API_KEY)"
        
        # Format features for the prompt
        feature_text = ", ".join([f"{feat['feature']} ({'+' if feat['value'] > 0 else ''}{feat['value']:.2f})" for feat in features[:5]])
        
        fraud_status = "FRAUD" if is_fraud else "LEGITIMATE"
        transaction_ref = f"Transaction #{transaction_id}" if transaction_id else "This transaction"
        
        prompt = f"""
        {transaction_ref} | Risk Score: {probability:.3f}
        The model classified this transaction as {fraud_status}.
        Key influential features: {feature_text}.
        
        Explain in simple business terms why this transaction was flagged as {fraud_status.lower()}.
        Focus on the most important risk factors and provide actionable insights.
        Keep the explanation concise and professional.
        """
        
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a fraud detection expert explaining transaction risk assessments to banking professionals."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"LLM justification error: {e}")
        return f"LLM explanation unavailable: {str(e)}"

def generate_explanation(features: List[Dict[str, Any]], is_fraud: bool, probability: float) -> str:
    """Generate human-readable explanation for fraud prediction"""
    if is_fraud:
        risk_factors = []
        for feature in features[:3]:  # Top 3 features
            name = feature['feature']
            value = feature['value']
            importance = feature['importance']
            
            if 'velocity' in name.lower():
                risk_factors.append(f"unusually high transaction velocity ({name}: {value:.1f})")
            elif 'fraud' in name.lower():
                risk_factors.append(f"device has previous fraud history ({value} cases)")
            elif 'credit' in name.lower() and 'risk' in name.lower():
                risk_factors.append(f"low credit risk score ({value})")
            elif 'session' in name.lower():
                risk_factors.append(f"suspicious session duration ({value:.1f} minutes)")
            elif 'bank' in name.lower() and 'months' in name.lower():
                risk_factors.append(f"new customer with limited history ({value} months)")
        
        if risk_factors:
            return f"High risk transaction detected. Key factors: {', '.join(risk_factors)}."
        else:
            return f"High risk transaction detected with {probability:.1%} probability."
    else:
        return "Low risk transaction. Normal customer patterns and transaction characteristics detected."

# =====================
# API Endpoints
# =====================
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🚀 Starting FraudLens API...")
    success = load_models()
    if not success:
        print("❌ Failed to load models. API may not work correctly.")
    else:
        print("✅ API ready to serve requests!")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FraudLens API - Advanced Fraud Detection",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = len(models) > 0 and len(preprocessing) > 0
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
        uptime=0.0,  # You can implement proper uptime tracking
        timestamp=datetime.now().isoformat()
    )

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": list(models.keys()),
        "preprocessing_loaded": len(preprocessing) > 0,
        "model_info": {
            "type": "LightGBM",
            "features": 31,
            "version": "1.0.0"
        }
    }

@app.post("/predict", response_model=FraudPrediction)
async def predict_single(transaction: TransactionData):
    """Predict fraud for a single transaction"""
    try:
        if not models or not preprocessing:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction.model_dump()])
        
        # Preprocess
        X = preprocess_data(df)
        
        # Predict
        model = models['final']
        probability = model.predict_proba(X)[0][1]
        is_fraud = probability > 0.5
        confidence = abs(probability - 0.5) * 2
        
        # Generate feature importance (simplified)
        feature_names = preprocessing.get("num_cols", []) + preprocessing.get("cat_cols", [])
        top_features = [
            {"feature": "velocity_6h", "importance": 0.18, "value": transaction.velocity_6h},
            {"feature": "device_fraud_count", "importance": 0.12, "value": transaction.device_fraud_count},
            {"feature": "credit_risk_score", "importance": 0.11, "value": transaction.credit_risk_score},
            {"feature": "session_length_in_minutes", "importance": 0.09, "value": transaction.session_length_in_minutes},
            {"feature": "bank_months_count", "importance": 0.08, "value": transaction.bank_months_count}
        ]
        
        # Generate LLM explanation
        explanation = generate_llm_justification(top_features, is_fraud, probability, f"TXN-{hash(str(transaction.model_dump())) % 10000:04d}")
        
        return FraudPrediction(
            is_fraud=bool(is_fraud),
            fraud_probability=float(probability),
            risk_score=float(probability),
            confidence=float(confidence),
            explanation=explanation,
            top_features=convert_numpy_types(top_features)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict fraud for multiple transactions"""
    try:
        if not models or not preprocessing:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        start_time = datetime.now()
        
        # Convert to DataFrame
        transactions_data = [t.model_dump() for t in request.transactions]
        df = pd.DataFrame(transactions_data)
        
        # Preprocess
        X = preprocess_data(df)
        
        # Predict
        model = models['final']
        probabilities = model.predict_proba(X)[:, 1]
        predictions = probabilities > 0.5
        
        # Generate predictions with explanations
        results = []
        fraud_count = 0
        
        for i, (prob, is_fraud) in enumerate(zip(probabilities, predictions)):
            if is_fraud:
                fraud_count += 1
            
            confidence = abs(prob - 0.5) * 2
            
            # Generate feature importance for this transaction
            top_features = [
                {"feature": "velocity_6h", "importance": 0.18, "value": df.iloc[i]['velocity_6h']},
                {"feature": "device_fraud_count", "importance": 0.12, "value": df.iloc[i]['device_fraud_count']},
                {"feature": "credit_risk_score", "importance": 0.11, "value": df.iloc[i]['credit_risk_score']},
                {"feature": "session_length_in_minutes", "importance": 0.09, "value": df.iloc[i]['session_length_in_minutes']},
                {"feature": "bank_months_count", "importance": 0.08, "value": df.iloc[i]['bank_months_count']}
            ]
            
            # Generate LLM explanation if requested
            if request.include_explanations:
                explanation = generate_llm_justification(top_features, is_fraud, prob, f"TXN-{i+1:04d}")
            else:
                explanation = ""
            
            results.append(FraudPrediction(
                is_fraud=bool(is_fraud),
                fraud_probability=float(prob),
                risk_score=float(prob),
                confidence=float(confidence),
                explanation=explanation,
                top_features=convert_numpy_types(top_features)
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Model metrics (mock for now)
        model_metrics = {
            "accuracy": 0.9016,
            "precision": 0.2459,
            "recall": 0.8586,
            "f1_score": 0.3823,
            "roc_auc": 0.9465
        }
        
        return BatchPredictionResponse(
            predictions=results,
            model_metrics=model_metrics,
            processing_time=processing_time,
            total_records=len(transactions_data),
            fraud_count=fraud_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload CSV file and analyze for fraud"""
    try:
        if not models or not preprocessing:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Check file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate required columns (basic check)
        required_columns = ['customer_age', 'velocity_6h', 'credit_risk_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Convert to TransactionData format
        transactions = []
        for _, row in df.iterrows():
            try:
                # Fill missing values with defaults
                transaction_data = {
                    'customer_age': int(row.get('customer_age', 30)),
                    'income': float(row.get('income', 0.5)),
                    'name_email_similarity': float(row.get('name_email_similarity', 0.5)),
                    'prev_address_months_count': int(row.get('prev_address_months_count', -1)),
                    'current_address_months_count': int(row.get('current_address_months_count', 12)),
                    'days_since_request': float(row.get('days_since_request', 0)),
                    'intended_balcon_amount': float(row.get('intended_balcon_amount', 0)),
                    'proposed_credit_limit': float(row.get('proposed_credit_limit', 1000)),
                    'payment_type': str(row.get('payment_type', 'AA')),
                    'bank_months_count': int(row.get('bank_months_count', 12)),
                    'has_other_cards': int(row.get('has_other_cards', 1)),
                    'foreign_request': int(row.get('foreign_request', 0)),
                    'velocity_6h': float(row.get('velocity_6h', 0)),
                    'velocity_24h': float(row.get('velocity_24h', 0)),
                    'velocity_4w': float(row.get('velocity_4w', 0)),
                    'zip_count_4w': int(row.get('zip_count_4w', 1)),
                    'bank_branch_count_8w': int(row.get('bank_branch_count_8w', 1)),
                    'date_of_birth_distinct_emails_4w': int(row.get('date_of_birth_distinct_emails_4w', 1)),
                    'credit_risk_score': int(row.get('credit_risk_score', 100)),
                    'employment_status': str(row.get('employment_status', 'CA')),
                    'housing_status': str(row.get('housing_status', 'BA')),
                    'email_is_free': int(row.get('email_is_free', 0)),
                    'phone_home_valid': int(row.get('phone_home_valid', 1)),
                    'phone_mobile_valid': int(row.get('phone_mobile_valid', 1)),
                    'source': str(row.get('source', 'INTERNET')),
                    'session_length_in_minutes': float(row.get('session_length_in_minutes', 5)),
                    'device_os': str(row.get('device_os', 'windows')),
                    'keep_alive_session': int(row.get('keep_alive_session', 1)),
                    'device_distinct_emails_8w': int(row.get('device_distinct_emails_8w', 1)),
                    'device_fraud_count': int(row.get('device_fraud_count', 0)),
                    'month': int(row.get('month', 0)),
                    'fraud_bool': int(row.get('fraud_bool', 0)) if 'fraud_bool' in row else None
                }
                transactions.append(TransactionData(**transaction_data))
            except Exception as e:
                print(f"Warning: Skipping invalid row: {e}")
                continue
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No valid transactions found in CSV")
        
        # Use batch prediction without LLM explanations for faster processing
        batch_request = BatchPredictionRequest(
            transactions=transactions,
            include_explanations=False
        )
        
        result = await predict_batch(batch_request)
        
        return {
            "filename": file.filename,
            "file_size": len(content),
            "results": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def generate_explanation(request: ExplanationRequest):
    """Generate LLM explanation for a specific transaction"""
    try:
        if not models or not preprocessing:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        start_time = datetime.now()
        
        # Generate feature importance for this transaction
        top_features = [
            {"feature": "velocity_6h", "importance": 0.18, "value": request.transaction_data.velocity_6h},
            {"feature": "device_fraud_count", "importance": 0.12, "value": request.transaction_data.device_fraud_count},
            {"feature": "credit_risk_score", "importance": 0.11, "value": request.transaction_data.credit_risk_score},
            {"feature": "session_length_in_minutes", "importance": 0.09, "value": request.transaction_data.session_length_in_minutes},
            {"feature": "bank_months_count", "importance": 0.08, "value": request.transaction_data.bank_months_count}
        ]
        
        # Generate LLM explanation
        explanation = generate_llm_justification(
            top_features, 
            request.prediction.is_fraud, 
            request.prediction.fraud_probability, 
            f"TXN-{hash(str(request.transaction_data.model_dump())) % 10000:04d}"
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ExplanationResponse(
            explanation=explanation,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation error: {str(e)}")

# =====================
# Run the API
# =====================
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
