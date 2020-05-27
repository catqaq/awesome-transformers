#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:44:53 2020

@author: jjg
"""

import torch
from transformers import BertTokenizer, Model2Model #Model2Model was deleted by "Delete Model2Model #3019"

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode the input to the encoder (the question)
question = "Who was Jim Henson?"
encoded_question = tokenizer.encode(question)

# Encode the input to the decoder (the answer)
answer = "Jim Henson was a puppeteer"
encoded_answer = tokenizer.encode(answer)

# Convert inputs to PyTorch tensors
question_tensor = torch.tensor([encoded_question])
answer_tensor = torch.tensor([encoded_answer])

# In order to compute the loss we need to provide language model
# labels (the token ids that the model should have produced) to
# the decoder.
lm_labels =  encoded_answer
labels_tensor = torch.tensor([lm_labels])

# Load pre-trained model (weights)
model = Model2Model.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# If you have a GPU, put everything on cuda
question_tensor = question_tensor.to('cuda')
answer_tensor = answer_tensor.to('cuda')
labels_tensor = labels_tensor.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(question_tensor, answer_tensor, decoder_lm_labels=labels_tensor)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the value of the LM loss 
    lm_loss = outputs[0]


# Let's re-use the previous question
question = "Who was Jim Henson?"
encoded_question = tokenizer.encode(question)
question_tensor = torch.tensor([encoded_question])

# This time we try to generate the answer, so we start with an empty sequence
answer = "[CLS]"
encoded_answer = tokenizer.encode(answer, add_special_tokens=False)
answer_tensor = torch.tensor([encoded_answer])

# Load pre-trained model (weights)
model = Model2Model.from_pretrained('fine-tuned-weights')
model.eval()

# If you have a GPU, put everything on cuda
question_tensor = question_tensor.to('cuda')
answer_tensor = answer_tensor.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(question_tensor, answer_tensor)
    predictions = outputs[0]

# confirm we were able to predict 'jim'
predicted_index = torch.argmax(predictions[0, -1]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'jim'