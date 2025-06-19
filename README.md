# Hugging Face Question & Answer Fine Tuning Example


## Steps to Use This Model and API
- ```pip install -r requirements.txt```
- ```python train.py``` This will train the model and place the trained model data in the qa-final-model/ directory. Wait for training to complete.
- In the terminal enter: ```uvicorn api:app --reload``` This will serve the API.
- Use the cURL below to make a request to the API.


### Results From Last Training

```
{
  'eval_loss': 1.2156516313552856,
  'eval_exact_match': 77.22800378429517,
  'eval_f1': 85.52556319548594,
  'eval_runtime': 46.5439,
  'eval_samples_per_second': 231.695,
  'eval_steps_per_second': 28.962,
  'epoch': 3.0
}            
{
  'train_runtime': 3703.7468,
  'train_samples_per_second': 71.704,
  'train_steps_per_second': 8.963,
  'train_loss': 1.0161355514613697,
  'epoch': 3.0
}
```

### Sample Postman Request

```
curl --location 'http://127.0.0.1:8000/ask' \
--header 'Content-Type: application/json' \
--data '{
    "question": "Where is the majority of the Amazon rainforest located?",
    "context": "The Amazon rainforest, often referred to as the lungs of the planet, is a vast tropical rainforest in South America. It covers over 5.5 million square kilometers, with the majority of it located in Brazil. The rainforest is home to an incredibly diverse range of species, including more than 2.5 million insect species, tens of thousands of plants, and around 2,000 birds and mammals. It also plays a vital role in regulating the Earth'\''s oxygen and carbon dioxide levels."
}'
```