from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
# model_path = "app/model"
model_path = r"app/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Intent map (must match your training!)
id2intent = {
    0: "missed_dose",
    1: "side_effect",
    2: "general_query",
    3: "acknowledgement"
}

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_intent(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return {"intent": id2intent[predicted_label]}
