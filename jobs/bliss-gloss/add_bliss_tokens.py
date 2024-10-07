# Usage:
# python add_bliss_tokens.py data/bliss_gloss_cleaned.json ~/Downloads/LlamaBliss ./data/bliss_ids_added.json ./data/bliss_ids_not_added.json

import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# When True, assign output embeddings to new Bliss tokens so the model can output new Bliss tokens.
# When False, do not assign output embeddings to new tokens so the output layer is not aware of new tokens
# and the model only outputs English.
ASSIGN_BLISS_OUTPUT_EMBEDDING = False

# When True, zero out all old output embeddings and only keep the new Bliss embeddings
# so the model will only output Bliss tokens.
# When False, keep both old and new output embeddings so the model will output a mix of English and Bliss.
ZERO_OUT_OLD_OUTPUT_EMBEDDING = False

# When True, assign the new Bliss token with input and output embeddings of the first single-token gloss.
# When False, assign the mean input and output embeddings of all single-token glosses for each Bliss symbol.
USE_FIRST_SINGLE_TOKEN_GLOSS = True

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
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)


def preprocess_gloss(gloss):
    return f" {gloss.replace(' ', '')}"


# Loop through the given glosses and return the first gloss that can be converted into
# a single token, along with its input and output embeddings. If none of the glosses
# can be converted to a single token, return (None, None, None)
# Return: a tuple of (first_single_token_gloss, input_embedding, output_embedding)
def get_first_single_token_embeddings(model, tokenizer, glosses):
    single_token_gloss = None
    for gloss in glosses:
        gloss = preprocess_gloss(gloss)
        tokens = tokenizer.tokenize(gloss)
        token_id = tokenizer.convert_tokens_to_ids(tokens)
        if len(token_id) == 1:
            single_token_gloss = gloss
            break

    if single_token_gloss:
        return single_token_gloss, model.get_input_embeddings().weight[token_id], model.lm_head.weight[token_id]
    else:
        return None, None, None


# Loop through the given glosses and return the first gloss that can be converted into
# a single token, along with its input and output embeddings. If none of the glosses
# can be converted to a single token, return (None, None, None)
# Return: a tuple of (first_single_token_gloss, input_embedding, output_embedding)
def get_mean_single_token_embeddings(model, tokenizer, glosses):
    print(f"glosses: {glosses}")
    single_token_glosses = []
    input_embeddings = []
    output_embeddings = []

    # Loop through each gloss for the current ID
    for gloss in glosses:
        gloss = preprocess_gloss(gloss)
        tokens = tokenizer.tokenize(gloss)
        token_id = tokenizer.convert_tokens_to_ids(tokens)

        # Check if it's a single token gloss
        if len(token_id) > 1:
            continue

        # Get input and output embedding of the token
        input_embedding = model.get_input_embeddings().weight[token_id]  # the input embedding
        output_embedding = model.lm_head.weight[token_id]  # the output embedding

        single_token_glosses.append(gloss)
        input_embeddings.append(input_embedding)
        output_embeddings.append(output_embedding)

    if len(single_token_glosses) > 0:
        print(f"single token glosses: {single_token_glosses}")
        mean_input_embedding = torch.mean(torch.stack(input_embeddings), dim=0)
        mean_output_embedding = torch.mean(torch.stack(output_embeddings), dim=0)

        return single_token_glosses, mean_input_embedding, mean_output_embedding
    else:
        return None, None, None


not_added_bliss_ids = {}
added_bliss_ids = {}
new_tokens_to_add = []
new_token_input_embeddings = []
new_token_output_embeddings = []

# Process each record in the JSON
for bliss_id, glosses in input_gloss_data.items():
    calculated_glosses, input_emb, output_emb = (get_first_single_token_embeddings if USE_FIRST_SINGLE_TOKEN_GLOSS else get_mean_single_token_embeddings)(model, tokenizer, glosses)

    if calculated_glosses:
        special_token = f"[BLISS_{bliss_id}]"
        new_tokens_to_add.append(special_token)
        new_token_input_embeddings.append(input_emb)
        if ASSIGN_BLISS_OUTPUT_EMBEDDING:
            new_token_output_embeddings.append(output_emb)
        added_bliss_ids[bliss_id] = calculated_glosses
    else:
        not_added_bliss_ids[bliss_id] = glosses

# Add all new tokens at once
num_added_bliss_ids = tokenizer.add_tokens(new_tokens_to_add)

# Resize token embeddings once
model.resize_token_embeddings(len(tokenizer))

# Update embeddings for new tokens
with torch.no_grad():
    if ZERO_OUT_OLD_OUTPUT_EMBEDDING:
        # Zero out all tokens in the output layer
        model.lm_head.weight.data.fill_(0)

    # Add new Bliss tokens
    for i, new_token in enumerate(new_tokens_to_add):
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        model.get_input_embeddings().weight[new_token_id] = new_token_input_embeddings[i]
        if ASSIGN_BLISS_OUTPUT_EMBEDDING:
            model.lm_head.weight[new_token_id] = new_token_output_embeddings[i]

print("\n")
print(f"{len(added_bliss_ids)} are added.")
print(f"{len(not_added_bliss_ids)} are not added.")

# # Save the updated model
# print("Saving updated model...")
# model.save_pretrained(output_model_path)
# tokenizer.save_pretrained(output_model_path)
# print("Model saved...")

# Save added tokens to file
with open(output_file_with_added_id, 'w') as f:
    json.dump(added_bliss_ids, f, indent=2)

# Save not added tokens to file
with open(output_file_with_not_added_id, 'w') as f:
    json.dump(not_added_bliss_ids, f, indent=2)


# Return next word predictions with probability greater than a threshold
def predict_next_words_above_threshold(sentence, threshold=0.1, max_predictions=10):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Generate logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the probabilities for the next token
    next_token_logits = logits[:, -1, :]  # Only the last token is relevant
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Filter tokens with probability above the threshold
    top_tokens = torch.where(probabilities > threshold)[1]
    top_probs = probabilities[0, top_tokens]

    # Convert token ids back to words
    top_words = [tokenizer.decode([token_id]) for token_id in top_tokens]

    # Sort by probability
    top_predictions = sorted(zip(top_words, top_probs), key=lambda x: -x[1])

    return top_predictions[:max_predictions]


# Return top_k number of next word predictions
def predict_top_k_words(sentence, top_k=10):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Generate logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the probabilities for the next token
    next_token_logits = logits[:, -1, :]  # Only the last token is relevant
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get the top n tokens with highest probabilities
    top_probs, top_tokens = torch.topk(probabilities, top_k)

    # Convert token ids back to words
    top_words = [tokenizer.decode([token_id]) for token_id in top_tokens[0]]

    top_predictions = list(zip(top_words, top_probs[0]))

    return top_predictions


# Convert bag of words into a sentence using the model. This function runs for CPU.
def generate_text_with_prompt_cpu(prompt, model, tokenizer, temperature=0.7):
    print(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to("cpu")
    attention_mask = inputs.attention_mask.to("cpu")

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=temperature
    )
    return tokenizer.batch_decode(outputs.cpu().detach().numpy(), skip_special_tokens=True)[0][len(prompt):]


# Convert bag of words into a sentence using the model. This function runs for GPU.
def generate_text_with_prompt_gpu(prompt, model, tokenizer, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    print(f"Prompt: {prompt}\n")
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=temperature)
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]


def convert_bag_of_words(model, tokenizer, bag_of_words):
    # Create a prompt to guide the model
    prompt = f"Create a complete sentence from the following words: {bag_of_words}."

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the sentence with the model
    output_tokens = model.generate(
        input_ids=inputs['input_ids'],
        max_length=100,  # limit the length of generated text
        num_beams=5,  # use beam search for better results
        no_repeat_ngram_size=2,  # prevent repetitive phrases
        early_stopping=True  # stop when a sentence is formed
    )

    # Decode the output tokens back to a string
    generated_sentence = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return generated_sentence


# sentences = ["I'm hungry so I want to", "[BLISS_14916][BLISS_14912][BLISS_14916][BLISS_18035][BLISS_17739]"]  # Incomplete sentence to predict
# top_k = 6  # Number of top predictions to return
# threshold = 0.1  # 10% probability threshold

# for sentence in sentences:
#     print(f"Predicted sentence: {sentence}")

#     predictions = predict_top_k_words(sentence, top_k)

#     print(f"Top {top_k} predictions:")
#     for word, prob in predictions:
#         print(f"{word}: {prob.item():.4f}")

# predictions = predict_next_words_above_threshold(probabilities, threshold)

# print(f"\nPredictions above threshold {threshold}:")
# for word, prob in predictions:
#     print(f"{word}: {prob.item():.4f}\n")

bag_of_words = ["nice weather walk", "[BLISS_15717][BLISS_18214][BLISS_18031]",
                "caregiver action keep home clean", "[BLISS_23063][BLISS_23007][BLISS_15143][BLISS_14885][BLISS_24062]"]

for words in bag_of_words:
    prompt = f"The given bag of words are from an AAC user who expresses himself telegraphically.\
 Please help to convert what the user said to first-person sentences. Only respond with converted sentences: {words}."
    print(f"A bag of word: {words}")
    # print(f"Converted sentence: {generate_text_with_prompt_gpu(prompt, model, tokenizer)}\n")
    print(f"Converted sentence: {generate_text_with_prompt_cpu(prompt, model, tokenizer)}\n")
