
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import os
import yaml

# 添加src到路径
import sys
sys.path.append('src')

from src.task.mtask.modelTrainTask import CreditModel

# 创建FastAPI应用
app = FastAPI(
    title="Credit Model API",
    description="API for credit scoring model",
    version="1.0.0"
)

# 加载配置
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载模型
model_path = config['paths']['model_output']
if os.path.exists(model_path):
    model = CreditModel.load_model(model_path)
    model_loaded = True
    print("Model loaded successfully")
else:
    model_loaded = False
    print("Warning: Model not found. API will not work until model is trained and saved.")


class CreditRequest(BaseModel):
    """
    信用评分请求数据模型
    """
    feature_1: float
    feature_2: float
    # 可以根据实际特征添加更多字段


class CreditResponse(BaseModel):
    """
    信用评分响应数据模型
    """
    prediction: int  # 0 or 1
    probability: float  # 0-1
    risk_level: str  # low, medium, high


class BatchCreditRequest(BaseModel):
    """
    批量信用评分请求数据模型
    """
    data: List[Dict[str, Any]]


class BatchCreditResponse(BaseModel):
    """
    批量信用评分响应数据模型
    """
    results: List[CreditResponse]


@app.get('/')
def read_root():
    return {"message": "Credit Model API", "status": "running"}


@app.get('/health')
def health_check():
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded
    }


@app.post('/predict', response_model=CreditResponse)
def predict_single(request: CreditRequest):
    """
    单条信用评分预测
    """
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 将请求数据转换为DataFrame
        input_data = pd.DataFrame([request.dict()])

        # 预测
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0])

        # 确定风险等级
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"

        return CreditResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post('/predict/batch', response_model=BatchCreditResponse)
def predict_batch(request: BatchCreditRequest):
    """
    批量信用评分预测
    """
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 将请求数据转换为DataFrame
        input_data = pd.DataFrame(request.data)

        # 预测
        predictions = model.predict(input_data).tolist()
        probabilities = model.predict_proba(input_data).tolist()

        results = []
        for pred, prob in zip(predictions, probabilities):
            # 确定风险等级
            if prob < 0.3:
                risk_level = "low"
            elif prob < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"

            results.append(
                CreditResponse(
                    prediction=int(pred),
                    probability=float(prob),
                    risk_level=risk_level
                )
            )

        return BatchCreditResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get('/model/info')
def get_model_info():
    """
    获取模型信息
    """
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    info = {
        "model_type": model.model_type,
        "feature_names": model.feature_names_,
        "best_params": model.best_params_ if hasattr(model, 'best_params_') else None
    }
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
