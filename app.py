from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import predict

app = FastAPI()

class TweetInput(BaseModel):
    input: str

@app.post("/predict")
def run_inference(data: TweetInput):
    try:
        result = predict(data.input)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
