# Multilingual Intent Classification API 🚀

A FastAPI-based REST API that uses `bert-base-multilingual-cased` fine-tuned on custom multilingual synthetic dataset to classify voice/text input into the following intents:

- `missed_dose`
- `side_effect`
- `general_query`
- `acknowledgment`

## 🌐 Supported Languages

English, Hindi, Gujarati, Marathi, Punjabi, Bengali, French, Spanish, German, Tamil, Telugu, Malayalam, Kannada

## ⚙️ Tech Stack

- Python
- FastAPI
- Transformers (HuggingFace)
- PyTorch
- Uvicorn
- BERT-base-multilingual-cased (Fine-tuned)

## 🚀 Run Locally

```bash
# Clone repo
git clone https://github.com/22AIML021VishalMehta/intent-classifier-api
cd intent-classifier-api

# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload
