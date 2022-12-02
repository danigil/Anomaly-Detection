import pickle

from fastapi import FastAPI
from pydantic import BaseModel

class Request(BaseModel):
    distance: int
    src_bytes: int
    dest_bytes: int

def load_model(model_name):
    model = pickle.load(open(model_name, 'rb'))
    return model

app = FastAPI()
model = load_model('isolation_forest_model.sav')

@app.get("/")
def default():
    return {"ok": 1}

@app.post("/")
def create_request(request: Request):
    distance = request.distance
    src_bytes = request.src_bytes
    dest_bytes = request.dest_bytes

    prediciton = model.predict([[distance, src_bytes, dest_bytes]])
    prediciton = list(map(lambda x: 1 if x == -1 else 0, prediciton))

    response = {
        "is_anomaly?" : prediciton[0]
    }
    return response

