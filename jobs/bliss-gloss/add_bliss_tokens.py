# Usage:
# python add_bliss_tokens.py data/bliss_gloss_cleaned.json ~/Downloads/LlamaBliss ./data/bliss_ids_added.json ./data/bliss_ids_not_added.json

import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if len(sys.argv) != 5:
    print("Usage: python add_bliss_tokens.py <input_gloss_json> <output_model_path> <output_file_with_added_id> <output_file_with_not_added_id>")
    sys.exit(1)

input_gloss_json = sys.argv[1]
output_model_path = sys.argv[2]
output_file_with_added_id = sys.argv[3]
output_file_with_not_added_id = sys.argv[4]

# Load the JSON file
with open(input_gloss_json, 'r') as f:
    input_gloss_data = json.load(f)

# Load the local Llama model
model_dir = os.path.expanduser("~") + "/Development/LLMs/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

ids_with_multiple_token_glosses = {}
added_tokens = {}
new_tokens_to_add = []
new_token_input_embeddings = []
new_token_output_embeddings = []

# Process each record in the JSON
for bliss_id, glosses in input_gloss_data.items():
    single_token_gloss = None
    for gloss in glosses:
        gloss = f" {gloss}"
        tokens = tokenizer.tokenize(gloss)
        token_id = tokenizer.convert_tokens_to_ids(tokens)
        if len(token_id) == 1:
            single_token_gloss = gloss
            break

    if single_token_gloss:
        special_token = f"[BLISS_{bliss_id}]"
        new_tokens_to_add.append(special_token)
        new_token_input_embeddings.append(model.get_input_embeddings().weight[token_id].clone())
        new_token_output_embeddings.append(model.lm_head.weight[token_id].clone())
        added_tokens[bliss_id] = single_token_gloss
        print(f"added: {bliss_id}, {single_token_gloss}")
    else:
        ids_with_multiple_token_glosses[bliss_id] = glosses
        print(f"not added: {bliss_id}, {glosses}")

# Add all new tokens at once
num_added_tokens = tokenizer.add_tokens(new_tokens_to_add)

# Resize token embeddings once
model.resize_token_embeddings(len(tokenizer))

# Update embeddings for new tokens
with torch.no_grad():
    for i, new_token in enumerate(new_tokens_to_add):
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        model.get_input_embeddings().weight[new_token_id] = new_token_input_embeddings[i]
        model.lm_head.weight[new_token_id] = new_token_output_embeddings[i]

print("\n")
print(f"{len(added_tokens)} are added.")
print(f"{len(ids_with_multiple_token_glosses)} are not added.")

# # Save the updated model
# print("Saving updated model...")
# model.save_pretrained(output_model_path)
# tokenizer.save_pretrained(output_model_path)
# print("Model saved...")

# Save added tokens to file
with open(output_file_with_added_id, 'w') as f:
    json.dump(added_tokens, f, indent=2)

# Save not added tokens to file
with open(output_file_with_not_added_id, 'w') as f:
    json.dump(ids_with_multiple_token_glosses, f, indent=2)


# Return next word predictions with probability greater than a threshold
def predict_next_words_above_threshold(sentence, threshold=0.1, max_predictions=10):
    # Filter tokens with probability above the threshold
    top_tokens = torch.where(probabilities > threshold)[1]
    top_probs = probabilities[0, top_tokens]

    # Convert token ids back to words
    top_words = [tokenizer.decode([token_id]) for token_id in top_tokens]

    # Sort by probability
    top_predictions = sorted(zip(top_words, top_probs), key=lambda x: -x[1])

    return top_predictions[:max_predictions]


# Return top_k number of next word predictions
def predict_top_k_words(probabilities, top_k=10):
    # Get the top n tokens with highest probabilities
    top_probs, top_tokens = torch.topk(probabilities, top_k)

    # Convert token ids back to words
    top_words = [tokenizer.decode([token_id]) for token_id in top_tokens[0]]

    top_predictions = list(zip(top_words, top_probs[0]))

    return top_predictions


sentence = "I'm hungry so I want to"  # Incomplete sentence to predict
top_k = 5  # Number of top predictions to return
threshold = 0.1  # 10% probability threshold

# Tokenize the input sentence
inputs = tokenizer(sentence, return_tensors="pt")

# Generate logits from the model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the probabilities for the next token
next_token_logits = logits[:, -1, :]  # Only the last token is relevant
probabilities = torch.softmax(next_token_logits, dim=-1)

# predictions = predict_top_k_words(probabilities, top_k)

# print(f"Top {top_k} predictions:")
# for word, prob in predictions:
#     print(f"{word}: {prob.item():.4f}")

predictions = predict_next_words_above_threshold(sentence, threshold)

print(f"\nPredictions above threshold {threshold}:")
for word, prob in predictions:
    print(f"{word}: {prob.item():.4f}")
