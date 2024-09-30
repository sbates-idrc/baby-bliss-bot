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

# Load the LLaMA 2 7B model and tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

ids_with_multiple_token_glosses = {}
added_tokens = {}
new_tokens_to_add = []
new_token_embeddings = []

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
        embedding = model.get_input_embeddings().weight[token_id].clone()
        new_token_embeddings.append(embedding)
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
        model.get_input_embeddings().weight[new_token_id] = new_token_embeddings[i]

print("\n")
print(f"{len(added_tokens)} are added.")
print(f"{len(ids_with_multiple_token_glosses)} are not added.")

# Query the model for next word prediction
input_text = "The quick brown fox jumps over the"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

print(f"Input: {input_text}")
print("Predictions:")

top_k = 10

with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0][:, -1, :]
    top_k_values, top_k_indices = torch.topk(predictions, top_k)

    next_token_id = top_k_indices[0][0].item()
    next_token = tokenizer.decode([next_token_id])

    print("  Next word:")
    for i in range(top_k):
        token = tokenizer.decode([top_k_indices[0][i].item()])
        probability = torch.softmax(top_k_values, dim=-1)[0][i].item()
        print(f"    {token} (probability: {probability:.4f})")

    input_ids = torch.cat([input_ids, top_k_indices[:, :1]], dim=-1)

generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(f"\nFull generated text: {generated_text}")

# Save the updated model
print("Saving updated model...")
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print("Model saved...")

# Save added tokens to file
with open(output_file_with_added_id, 'w') as f:
    json.dump(added_tokens, f, indent=2)

# Save not added tokens to file
with open(output_file_with_not_added_id, 'w') as f:
    json.dump(ids_with_multiple_token_glosses, f, indent=2)
