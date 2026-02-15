import os
import random
import asyncio
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pymongo import MongoClient

# ----------------------------
# App Initialization
# ----------------------------

app = FastAPI(title="Fraud Risk Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load ML Model
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "fraud_model.pkl")
model = joblib.load(model_path)

# ----------------------------
# MongoDB Connection
# ----------------------------

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["fraud_db"]
collection = db["transactions"]

# ----------------------------
# Request Schema
# ----------------------------

class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# ----------------------------
# Root Endpoint
# ----------------------------

@app.get("/")
def home():
    return {"message": "Fraud Risk Intelligence API Running"}

# ----------------------------
# Fraud Scoring Endpoint
# ----------------------------

@app.post("/score-transaction")
def score_transaction(transaction: TransactionInput):

    features = np.array(list(transaction.dict().values())).reshape(1, -1)

    probability = model.predict_proba(features)[0][1]

    if probability > 0.75:
        risk_level = "High"
        decision = "Block"
    elif probability > 0.40:
        risk_level = "Medium"
        decision = "Review"
    else:
        risk_level = "Low"
        decision = "Approve"

    feature_names = list(transaction.dict().keys())
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-3:]
    top_features = [feature_names[i] for i in top_indices]

    collection.insert_one({
        "input": transaction.dict(),
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "decision": decision,
        "top_risk_indicators": top_features,
        "timestamp": datetime.utcnow()
    })

    return {
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "decision": decision,
        "top_risk_indicators": top_features
    }

# ----------------------------
# Fetch All Transactions
# ----------------------------

@app.get("/transactions")
def get_transactions():
    data = list(collection.find({}, {"_id": 0}))
    return data

# ----------------------------
# Dashboard Summary Endpoint
# ----------------------------

@app.get("/dashboard")
def dashboard_summary():

    total = collection.count_documents({})

    high = collection.count_documents({"risk_level": "High"})
    medium = collection.count_documents({"risk_level": "Medium"})
    low = collection.count_documents({"risk_level": "Low"})

    approved = collection.count_documents({"decision": "Approve"})
    blocked = collection.count_documents({"decision": "Block"})
    review = collection.count_documents({"decision": "Review"})

    approval_rate = 0
    if total > 0:
        approval_rate = round((approved / total) * 100, 2)

    return {
        "total_transactions": total,
        "risk_distribution": {
            "high": high,
            "medium": medium,
            "low": low
        },
        "decision_distribution": {
            "approved": approved,
            "blocked": blocked,
            "review": review
        },
        "approval_rate_percent": approval_rate
    }

# ----------------------------
# AUTO TRANSACTION GENERATOR
# ----------------------------

def generate_random_transaction():
    return {
        "Time": random.uniform(0, 100000),
        "V1": random.uniform(-5, 5),
        "V2": random.uniform(-5, 5),
        "V3": random.uniform(-5, 5),
        "V4": random.uniform(-5, 5),
        "V5": random.uniform(-5, 5),
        "V6": random.uniform(-5, 5),
        "V7": random.uniform(-5, 5),
        "V8": random.uniform(-5, 5),
        "V9": random.uniform(-5, 5),
        "V10": random.uniform(-5, 5),
        "V11": random.uniform(-5, 5),
        "V12": random.uniform(-5, 5),
        "V13": random.uniform(-5, 5),
        "V14": random.uniform(-5, 5),
        "V15": random.uniform(-5, 5),
        "V16": random.uniform(-5, 5),
        "V17": random.uniform(-5, 5),
        "V18": random.uniform(-5, 5),
        "V19": random.uniform(-5, 5),
        "V20": random.uniform(-5, 5),
        "V21": random.uniform(-5, 5),
        "V22": random.uniform(-5, 5),
        "V23": random.uniform(-5, 5),
        "V24": random.uniform(-5, 5),
        "V25": random.uniform(-5, 5),
        "V26": random.uniform(-5, 5),
        "V27": random.uniform(-5, 5),
        "V28": random.uniform(-5, 5),
        "Amount": random.uniform(1, 10000)
    }

async def auto_generate_transactions():
    while True:

        transaction = generate_random_transaction()
        features = np.array(list(transaction.values())).reshape(1, -1)

        probability = model.predict_proba(features)[0][1]

        if probability > 0.75:
            risk_level = "High"
            decision = "Block"
        elif probability > 0.40:
            risk_level = "Medium"
            decision = "Review"
        else:
            risk_level = "Low"
            decision = "Approve"

        collection.insert_one({
            "input": transaction,
            "fraud_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "decision": decision,
            "timestamp": datetime.utcnow()
        })

        print("Auto transaction generated")

        await asyncio.sleep(5)

# ----------------------------
# START BACKGROUND TASK
# ----------------------------

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_generate_transactions())
