from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
qa_pipeline = pipeline('question-answering', model='./qa-final-model', tokenizer='./qa-final-model')

# Define the request body structure
class QAInput(BaseModel):
  question: str
  context: str

@app.post('/ask')
def ask(data: QAInput):
  result = qa_pipeline(question=data.question, context=data.context)
  return result