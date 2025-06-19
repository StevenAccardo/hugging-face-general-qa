from datasets import load_dataset

# Download the dataset
dataset = load_dataset('squad')
# print(dataset['train'][0])

{
  'id': '5733be284776f41900661182', # example_id later (this is how we track our chunks back to the original example)
  'title': 'University_of_Notre_Dame', # Not used
  'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
  'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
  'answers': {
    'text': ['Saint Bernadette Soubirous'],
    'answer_start': [515]
    }
}


from transformers import AutoTokenizer

# Download the tokenizer config for the checkpoint
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# We need to pre-process our data
def preprocess_function(examples):
    # Clean whitespace from questions
    questions = [question.strip() for question in examples['question']]

    # Tokenize question-context pairs with truncation and overflow handling
    inputs = tokenizer(
      questions,
      examples['context'], # Context is the information that the question is derived from, and where the answer is sourced as well.
      max_length=384, # Common max length for Q&A. 384-token inputs are ~30% faster than 512-token inputs and use less RAM.
      truncation='only_second', # Tells the tokenizer to truncate the 2nd 'sentecnce' which in our case our context. We don't want to truncate the question, rather the context.
      stride=128, # Used when return_overflowing_tokens=True. Dictates how many tokens should be overlapped for each subsequent chunk
      return_overflowing_tokens=True, # This tells the tokenizer to do the chunking.
      return_offsets_mapping=True # Returns a mapping from each token to its character position in the original string. This is essential for QA tasks where the model predicts a span of tokens — you need to convert that back to a character range to extract the answer string.
    )
    
    # inputs = {
    #   'input_ids': List[List[int]],            # Token IDs for each chunk
    #   'token_type_ids': List[List[int]],       # Segment IDs (0 for question, 1 for context), if the model uses them
    #   'attention_mask': List[List[int]],       # 1s for tokens, 0s for padding
    #   'offset_mapping': List[List[Tuple[int, int]]],  # Maps token → original char span (or None)
    #   'overflow_to_sample_mapping': List[int], # Index back to the original `examples` row
    #   'special_tokens_mask': List[List[int]],  # (optional) Mask indicating which tokens are special ([CLS], [SEP])
    # }

    # Track which tokenized chunk comes from which original example
    sample_mapping = inputs.pop('overflow_to_sample_mapping')
    offset_mapping = inputs['offset_mapping']
    inputs["start_positions"] = []
    inputs["end_positions"] = []
    inputs['example_id'] = []

    # For each tokenized window/chunk
    # Now that each record is chunked, the length will likely be longer than the original batch size
    for i in range(len(inputs['input_ids'])):
      # Get the index of the original record before chunking.
      sample_index = sample_mapping[i]
      answers = examples['answers'][sample_index]
      # We store the id of the original record in a new property on the dictionary
      # This is essential later during postprocessing, where we’ll group predictions by original example.
      inputs['example_id'].append(examples['id'][sample_index])

      # Keep offsets only for context tokens (sequence_id == 1)
      sequence_ids = inputs.sequence_ids(i)
      offset = inputs['offset_mapping'][i]
      inputs['offset_mapping'][i] = [
        # If the iteration below yields a sequence_id == 1, then replaces the offset_mappings with only the ones for the context(sequence==1)
        (offset_map if sequence_id == 1 else None)
        # Creates a list of tuples with 2 indexes in each tuple. The first being the offset_mapping and the other being the sequence_id
        # Then we can deconstruct the offset_map and sequence_id for each tuple in the list
        for offset_map, sequence_id in zip(offset, sequence_ids)
      ]
      
      # Handle possible no-answer case
      if len(answers['answer_start']) == 0:
        inputs['start_positions'].append(0)
        inputs['end_positions'].append(0)
      else:
        # Find the start and end indices of the answer in the context
        start_char_idx = answers['answer_start'][0]
        end_char_idx = start_char_idx + len(answers['text'][0])

        # Find the start and end token indices in the context
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
          token_start_index += 1

        token_end_index = len(inputs['input_ids'][i]) - 1
        while sequence_ids[token_end_index] != 1:
          token_end_index -= 1

        # Now find token indices that match the char positions
        while token_start_index <= token_end_index and offset[token_start_index] and offset[token_start_index][0] <= start_char_idx:
          token_start_index += 1
        token_start_index -= 1

        while token_end_index >= token_start_index and offset[token_end_index] and offset[token_end_index][1] >= end_char_idx:
          token_end_index -= 1
        token_end_index += 1

        inputs['start_positions'].append(token_start_index)
        inputs['end_positions'].append(token_end_index)

    return inputs

# Here is how we apply the tokenization function on all our datasets at once.
# We’re using batched=True in our call to map so the function is applied to multiple elements of our dataset at once, and not on each element separately.
# This allows for faster preprocessing.
# Removed the columns from that are not going to be used as input for the model, as they won't be useful. Keep only the outputs from the the preprocess/tokination invocation
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
# print(tokenized_dataset['train'][0])

{
  'input_ids': [101, 2000, 3183, 2106, 1996, 6261, 2984, 9382, 3711, 1999, 8517, 1999, 10223, 26371, 2605, 1029, 102, 6549, 2135, 1010, 1996, 2082, 2038, 1037, 3234, 2839, 1012, 10234, 1996, 2364, 2311, 1005, 1055, 2751, 8514, 2003, 1037, 3585, 6231, 1997, 1996, 6261, 2984, 1012, 3202, 1999, 2392, 1997, 1996, 2364, 2311, 1998, 5307, 2009, 1010, 2003, 1037, 6967, 6231, 1997, 4828, 2007, 2608, 2039, 14995, 6924, 2007, 1996, 5722, 1000, 2310, 3490, 2618, 4748, 2033, 18168, 5267, 1000, 1012, 2279, 2000, 1996, 2364, 2311, 2003, 1996, 13546, 1997, 1996, 6730, 2540, 1012, 3202, 2369, 1996, 13546, 2003, 1996, 24665, 23052, 1010, 1037, 14042, 2173, 1997, 7083, 1998, 9185, 1012, 2009, 2003, 1037, 15059, 1997, 1996, 24665, 23052, 2012, 10223, 26371, 1010, 2605, 2073, 1996, 6261, 2984, 22353, 2135, 2596, 2000, 3002, 16595, 9648, 4674, 2061, 12083, 9711, 2271, 1999, 8517, 1012, 2012, 1996, 2203, 1997, 1996, 2364, 3298, 1006, 1998, 1999, 1037, 3622, 2240, 2008, 8539, 2083, 1017, 11342, 1998, 1996, 2751, 8514, 1007, 1010, 2003, 1037, 3722, 1010, 2715, 2962, 6231, 1997, 2984, 1012, 102],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  'offset_mapping': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, [0, 13], [13, 15], [15, 16], [17, 20], [21, 27], [28, 31], [32, 33], [34, 42], [43, 52], [52, 53], [54, 58], [59, 62], [63, 67], [68, 76], [76, 77], [77, 78], [79, 83], [84, 88], [89, 91], [92, 93], [94, 100], [101, 107], [108, 110], [111, 114], [115, 121], [122, 126], [126, 127], [128, 139], [140, 142], [143, 148], [149, 151], [152, 155], [156, 160], [161, 169], [170, 173], [174, 180], [181, 183], [183, 184], [185, 187], [188, 189], [190, 196], [197, 203], [204, 206], [207, 213], [214, 218], [219, 223], [224, 226], [226, 229], [229, 232], [233, 237], [238, 241], [242, 248], [249, 250], [250, 252], [252, 254], [254, 256], [257, 259], [260, 262], [263, 265], [265, 268], [268, 269], [269, 270], [271, 275], [276, 278], [279, 282], [283, 287], [288, 296], [297, 299], [300, 303], [304, 312], [313, 315], [316, 319], [320, 326], [327, 332], [332, 333], [334, 345], [346, 352], [353, 356], [357, 365], [366, 368], [369, 372], [373, 375], [375, 379], [379, 380], [381, 382], [383, 389], [390, 395], [396, 398], [399, 405], [406, 409], [410, 420], [420, 421], [422, 424], [425, 427], [428, 429], [430, 437], [438, 440], [441, 444], [445, 447], [447, 451], [452, 454], [455, 458], [458, 462], [462, 463], [464, 470], [471, 476], [477, 480], [481, 487], [488, 492], [493, 500], [500, 502], [503, 511], [512, 514], [515, 520], [521, 525], [525, 528], [528, 531], [532, 534], [534, 536], [536, 539], [539, 541], [542, 544], [545, 549], [549, 550], [551, 553], [554, 557], [558, 561], [562, 564], [565, 568], [569, 573], [574, 579], [580, 581], [581, 584], [585, 587], [588, 589], [590, 596], [597, 601], [602, 606], [607, 615], [616, 623], [624, 625], [626, 633], [634, 637], [638, 641], [642, 646], [647, 651], [651, 652], [652, 653], [654, 656], [657, 658], [659, 665], [665, 666], [667, 673], [674, 679], [680, 686], [687, 689], [690, 694], [694, 695], None],
  'example_id': '5733be284776f41900661182' # Same id from the original dataset. How we can tie chunks back together after training
}

from transformers import AutoModelForQuestionAnswering

# We are using the pretrained model that is trained for mask filling. This will give us a good base to work off of, and allows the model to understand the inputs pretty well already.
# However, the base model is not trained on Q&A, and therefore we have to place a Q&A head on the model.
# The Q&A head will be instantiated with random weights, and will not perform Q&A well, so we will have to fine tune it on a new data model that is set up for Q&A. In order for it to perform well.
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased')


from transformers import TrainingArguments, Trainer
import collections
import numpy as np

# This is especially necessary in Question & Answer tasks
# 
def postprocess_qa_predictions(
    examples, # list of validation examples (dicts with 'id', 'context', 'question', etc.)
    features, # tokenized windows of each example with overflow and offset mappings. The data stored in the tokenized_dataset variable
    predictions, # tuple of (all_start_logits, all_end_logits) from the model output
    tokenizer,
    n_best_size=20, # An option that will tell the model how many of the top logit scores it should take into account when predicting the answer
    max_answer_length=30
):
  '''
  Converts raw start/end logits into final answer strings for each example.
  Handles overlapping contexts, token-to-text mapping, and scoring logic.

  Returns:
    A dict mapping example ID → predicted text answer.
  '''

  # Unpack model outputs: start and end logits for each feature
  all_start_logits, all_end_logits = predictions
  # Schema: all_start_logits -> List[List[float]]
  # all_start_logits = [
  #     [3.2, -1.1, 0.8, ...],   # logits for feature 0 (first chunk of an example)
  #     [2.9, -0.5, 1.2, ...],   # logits for feature 1 (next chunk of same or different example)
  #     ...
  # ]

  # Creates a dictionary that will create a new empty list as the value of a property if the key wasn't already in the dictionary
  features_per_example = collections.defaultdict(list)
  # Iterate through each of the feature chunks returning the index and the feature itself
  for i, feature in enumerate(features):
    # Create a new key in the dictionary for, if not already there, as the example_id. Then append the index of the feature to the list value.
    # We use this list later to aggregate the chunked features under each source example that they were created from.
    features_per_example[feature['example_id']].append(i)

  # Final answers go here (OrderedDict preserves input order)
  final_predictions = collections.OrderedDict()

  # Loop through each original example (not tokenized chunks)
  for example in examples:
    example_id = example['id']
    context = example['context']
    # This is what we just created above
    feature_indices = features_per_example[example_id]

    min_null_score = None  # Used to keep track of the 'no answer' score
    valid_answers = []     # Store valid predicted spans for this example

    # Loop through each feature (tokenized window) corresponding to this example
    for feature_index in feature_indices:
      start_logits = all_start_logits[feature_index]
      end_logits = all_end_logits[feature_index]
      offset_mapping = features[feature_index]['offset_mapping'] # This contains either None for special characters or the start and end indexes for each column in the context. A list of tuples and None.
      input_ids = features[feature_index]['input_ids']

      # Find index of [CLS] token
      # The start and end logits are often assigned to the cls token when the model doesn't believe there is an answer in the context chunk
      cls_index = input_ids.index(tokenizer.cls_token_id)

      # Null prediction score = sum of [CLS] token's start and end logits
      # If the model thinks that there is no answer in the chunk, then it will give a higher raw score to the CLS token
      # As we iterate through each chunk tied to the example, we take the lowest null_score to later be compared against potential answer span scores
      # This will later be compared when checking each chunk per example to tell if there is no answer in the context at all in which case this value will be the highest value, and therefore equates to no answer found
      null_score = start_logits[cls_index] + end_logits[cls_index]
      if min_null_score is None or null_score < min_null_score:
        min_null_score = null_score

      # Get indices of top `n_best_size` start and end logits
      # argsort defaults to ascending order. Passing the negative n_best_size will take the n_best_size indexes from the tail of the list
      start_indexes = np.argsort(start_logits)[-n_best_size:]
      end_indexes = np.argsort(end_logits)[-n_best_size:]

      # Try every combination of top start and end indices
      for start_index in start_indexes:
        for end_index in end_indexes:
          # Skip if out of range
          # Ensures the start_index and end_index being added together are part of this chunk by ensuring their indexes are within the bounds of 0 to the length fo the offset_mapping list
          if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
            continue
          # Skip if there's no char mapping (e.g., special tokens)
          if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
            continue
          # Skip if end is before start
          if end_index < start_index:
            continue
          # Skip if the predicted span is too long
          if (end_index - start_index + 1) > max_answer_length:
            continue
          
          # If the indexes made it passed the filter
          # Convert token indices to character positions in the original context
          start_char = offset_mapping[start_index][0] # (6, 15) -> 6
          end_char = offset_mapping[end_index][1] # (19, 25) -> 25
          predicted_text = context[start_char:end_char] # Returns the span of text from the start to the end index in the original context list

          # Store this valid answer candidate
          valid_answers.append({
            'score': start_logits[start_index] + end_logits[end_index],
            'text': predicted_text
          })

    # Select the best valid span based on score (or fallback to empty string). We will choose the one with the highest raw score
    if valid_answers:
      best_answer = max(valid_answers, key=lambda x: x['score'])
    else:
      best_answer = {'text': '', 'score': 0.0}

    # Save best prediction for this example
    final_predictions[example_id] = best_answer['text']

  return final_predictions

from evaluate import load
from transformers import EvalPrediction

metric = load('squad')

def compute_metrics(p: EvalPrediction):
    '''
    Custom compute_metrics function for Hugging Face Trainer.
    Converts model predictions (start/end logits) into answer text strings,
    and compares them to ground truth answers using SQuAD's exact match (EM) and F1 metrics.

    Arguments:
    - p: EvalPrediction object with:
        - p.predictions: Tuple of (start_logits, end_logits)
        - p.label_ids: Not used directly in QA (used in classification tasks)

    Returns:
    - Dictionary with 'exact_match' and 'f1' keys.
    '''
    # Run the postprocessing logic to turn raw logits into text answers
    final_predictions = postprocess_qa_predictions(
        examples=dataset['validation'], # Original validation examples (with ID, context, etc.)
        features=tokenized_dataset['validation'], # Tokenized + chunked features
        predictions=p.predictions, # (start_logits, end_logits)
        tokenizer=tokenizer
    )

    # Format predictions for the SQuAD metric
    # - Each prediction must be a dict with keys: 'id' and 'prediction_text'
    formatted_predictions = [
        {'id': k, 'prediction_text': v} for k, v in final_predictions.items()
    ]

    # Ground truth format: each item must have keys: 'id' and 'answers'
    # - 'answers' must contain a dict with 'text' and 'answer_start'
    references = [
        {'id': ex['id'], 'answers': ex['answers']} for ex in dataset['validation']
    ]

    # Compute exact match and F1 score
    return metric.compute(predictions=formatted_predictions, references=references)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
  output_dir='./training-output', # The output directory where the model predictions and checkpoints will be written.
  eval_strategy='epoch', # When to run evaluation: every 'epoch', 'steps', or 'no'
  save_strategy='epoch', # How often the model weights and etc are saved
  learning_rate=2e-5, # Base learning rate for the optimizer (AdamW by default)
  per_device_train_batch_size=8, # Batch size per GPU (or CPU) during training
  per_device_eval_batch_size=8, # Batch size per GPU (or CPU) during evaluation
  num_train_epochs=3, # Number of full passes through the training dataset (Default is 3)
  weight_decay=0.01,  # L2 regularization: helps prevent overfitting
  logging_dir='./training-logs', # Where to write logs (for TensorBoard or debugging)
  load_best_model_at_end=True,   # At end of training, load the checkpoint with best eval score
  metric_for_best_model='eval_f1', # Use F1 score as the key metric
  greater_is_better=True          # Higher F1 is better, so keep the highest
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset['train'],
  eval_dataset=tokenized_dataset['validation'],
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics
)

trainer.train()

# Uncomment for manual evaluation or testing with test dataset
# trainer.evaluate()

preds = trainer.predict(tokenized_dataset['validation'])
text_preds = postprocess_qa_predictions(
    dataset['validation'],
    tokenized_dataset['validation'],
    preds.predictions,
    tokenizer,
)
for i, (k, v) in enumerate(text_preds.items()):
    print(f'Question {i}: {dataset["validation"][i]["question"]}')
    print(f'Pred: {v}')
    print(f'GT  : {dataset["validation"][i]["answers"]["text"][0]}')
    if i == 4: break


trainer.save_model('./qa-final-model')
tokenizer.save_pretrained('./qa-final-model')