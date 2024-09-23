# Experiment to add two Bliss specific tokens to the Llama tokenizer by associating them
# with the embedding of word "light" and "bark"
# Issues:
# 1. If simply replacing "light" with "[BLISS_1]" in a sentence, the input embedding sequence
# for these two sentences have an offset of 1 in the tensor size. "[BLISS_1]" sentence has one
# more embedding than the "light" sentence. It's because of the parsing of the special token "_"
# used for identifying the word start:
#
# Phrase: She turned on the light in the room.
# tokens: ['▁She', '▁turned', '▁on', '▁the', '▁light', '▁in', '▁the', '▁room', '.']
# Phrase: She turned on the [BLISS_1] in the room.
# tokens: ['▁She', '▁turned', '▁on', '▁the', '▁', '[BLISS_1]', '▁in', '▁the', '▁room', '.']
#
# In order to get rid of the extra '_' embedding in the "[BLISS_1]" sentence, the input sentence
# needs to be:
#
# She turned on the[BLISS_1] in the room.
# This issue shows simply replacing a word with the custom string is not applicable as language model
# has its internal handling for word/sentence boudaries and special tokens.
#
# 2. Cannot assign the embedding of "bark" to a custom token because this word is translated to
# two tokens: ['▁b', 'ark'] that map to two embeddings, which cannot be assigned to a single custom token.

import os
from transformers import LlamaTokenizer, LlamaModel
# import torch

# Load the LLaMA 2 7B model and tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-2-7b-hf"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaModel.from_pretrained(model_dir)

# Add special tokens
special_tokens = ["[BLISS_1]"]
# special_tokens = ["[BLISS_1]", "[BLISS_2]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))

# Set embeddings for special tokens
word_to_token = {"light": "[BLISS_1]"}
# word_to_token = {"light": "[BLISS_1]", "bark": "[BLISS_2]"}

for word, special_token in word_to_token.items():
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    word_ids = tokenizer.encode(word, add_special_tokens=False)
    print("tokens: ", tokens)
    print("token_ids: ", token_ids)
    print("word_ids: ", word_ids)
    # print("===\n")

    # word_ids = tokenizer.encode(word, add_special_tokens=False)
    if len(word_ids) > 1:
        print(f"Warning: '{word}' is tokenized into multiple tokens. Using the first token's embedding.")
    word_id = word_ids[0]
    word_embedding = model.get_input_embeddings().weight[word_id].clone()
    # word_embedding = model.get_input_embeddings()(tokenizer(word, return_tensors="pt")["input_ids"][0])
    special_token_index = tokenizer.convert_tokens_to_ids(special_token)
    print(f"Shape of embedding for '{word}': {word_embedding.shape}")
    print(f"special_token_index: {special_token_index}")
    model.get_input_embeddings().weight.data[special_token_index] = word_embedding


def replace_word_with_special_token(phrase, word, special_token, tokenizer):
    # Tokenize the original phrase
    original_tokens = tokenizer.tokenize(phrase)

    # Find the index of the word to replace
    word_index = original_tokens.index(f"▁{word}")

    # Replace the word with the special token
    original_tokens[word_index] = special_token

    # Join the tokens back into a string
    new_phrase = tokenizer.convert_tokens_to_string(original_tokens)

    return new_phrase


# Example usage
original_phrase = "She turned on the light in the room."
new_phrase = replace_word_with_special_token(original_phrase, "light", "[BLISS_1]", tokenizer)
print(f"NEW: {new_phrase}")

# # Define phrases for comparison
# phrases = {
#     "light": [
#         "She turned on the light in the room.",
#         "She prefers to eat a light meal in the evening."
#     # ],
#     # "bark": [
#     #     "The dog began to bark loudly at the stranger.",
#     #     "The tree's bark was rough and covered in moss."
#     ]
# }

# # Function to get embeddings for a phrase
# def get_phrase_embedding(tokenizer, model, phrase):
#     print(f"\nPhrase: {phrase}")
#     tokens = tokenizer.tokenize(phrase)
#     inputs = tokenizer(phrase, return_tensors="pt")
#     print(f"\ntokens: {tokens}")
#     print(f"\ntoken inputs: {inputs}")
#     with torch.no_grad():
#         outputs = model(**inputs)

#     input_embedding = model.embed_tokens(inputs['input_ids'])
#     output_embedding = outputs.last_hidden_state.mean(dim=1)
#     print(f"Shape of input_embedding: {input_embedding.shape}")
#     print(f"Shape of output_embedding: {output_embedding.shape}\n")
#     return {
#         "input_embedding": input_embedding,
#         "output_embedding": output_embedding
#     }

# # Perform comparisons
# for word, token in word_to_token.items():
#     print(f"Comparing '{word}' with '{token}':")
#     print("-" * 50)

#     for phrase in phrases[word]:
#         original_phrase = phrase
#         special_phrase = phrase.replace(word, token)

#         original_embedding = get_phrase_embedding(tokenizer, model, original_phrase)
#         special_embedding = get_phrase_embedding(tokenizer, model, special_phrase)

#         # similarity_input_embedding = torch.nn.functional.cosine_similarity(original_embedding["input_embedding"], special_embedding["input_embedding"])
#         # similarity_output_embedding = torch.nn.functional.cosine_similarity(original_embedding["output_embedding"], special_embedding["output_embedding"])

#         # print(f"\nOriginal: {original_phrase}")
#         # print(f"Special:  {special_phrase}")
#         # print(f"Input embedding: Are the embeddings exactly the same? {torch.allclose(original_embedding['input_embedding'], special_embedding['input_embedding'])}")
#         # print(f"Input embedding: Cosine similarity between embeddings: {similarity_input_embedding:.4f}")
#         # print(f"Output embedding: Are the embeddings exactly the same? {torch.allclose(original_embedding['output_embedding'], special_embedding['output_embedding'])}")
#         # print(f"Output embedding: Cosine similarity between embeddings: {similarity_output_embedding:.4f}")
