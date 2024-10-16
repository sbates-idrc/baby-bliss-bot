# Compare input embeddings and output embeddings of the same word in difference sentences(contexts)
# 1. The input embedding to the first layer of the neural netword
# 2. The output embedding from the last layer of the neural netword

import os
from transformers import LlamaTokenizer, LlamaModel
import torch
from utility_funcs import compare_embedding

# The local directory with the model and the tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-2-7b-hf"

# Define phrases and the word in them for looking up embeddings
examples = [
  {
    "phrases": ["She turned on the light in the room.", "She prefers to eat a light meal in the evening."],
    "word": "light"
  },
  {
    "phrases": ["The dog began to bark loudly at the stranger.", "The tree’s bark was rough and covered in moss."],
    "word": "bark"
  },
  {
    "phrases": ["He deposited his paycheck at the bank.", "They had a picnic by the river bank."],
    "word": "bank"
  },
  {
    "phrases": ["She decided to run five miles this morning.", "He will run in the upcoming marathon."],
    "word": "run"
  },
  {
    "phrases": ["He is a strong swimmer and can cross the river easily.", "Her argument was strong and convinced everyone in the room."],
    "word": "strong"
  },
  {
    "phrases": ["The weather outside is very cold today.", "She caught a cold and has been sneezing all day."],
    "word": "cold"
  }
]

# Initialize the Llama tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaModel.from_pretrained(model_dir)


# Function to get embeddings for a specific word in a phrase
def get_word_embedding(phrase, word):
    # Tokenize the phrase
    inputs = tokenizer(phrase, return_tensors="pt")

    # Get the token ID for the target word
    word_token_id = tokenizer.encode(word, add_special_tokens=False)[0]
    print(f"Token ID for '{word}' in phrase '{phrase}': {word_token_id}")

    # Find the position of the word in the tokenized input
    word_position = (inputs.input_ids == word_token_id).nonzero()[0, 1]
    print(f"Word position for '{word} in phrase '{phrase}': {word_position}")

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the input embeddings of the phrase
    input_embeddings = model.embed_tokens(inputs['input_ids'])
    print(f"inputs['input_ids']: {inputs['input_ids']}")

    return {
        "input_embedding": input_embeddings[0, word_position],
        "output_embedding": outputs.last_hidden_state[0, word_position, :]
    }


for i, example in enumerate(examples, 1):
    print(f"\nExample {i}: Word '{example['word']}'")
    print("-" * 50)

    print(f"Phrase 1: '{example['phrases'][0]}'")
    print(f"Phrase 2: '{example['phrases'][1]}'")

    # Get embeddings for the word in both phrases
    embeddings = [get_word_embedding(phrase, example['word']) for phrase in example['phrases']]

    # Compare the embeddings
    input_result = compare_embedding(embeddings[0]["input_embedding"], embeddings[1]["input_embedding"])
    output_result = compare_embedding(embeddings[0]["output_embedding"], embeddings[1]["output_embedding"])

    # Print results
    print(f"Input embedding: Are the embeddings exactly the same? {input_result['are_equal']}")
    print(f"Input embedding: Cosine similarity between embeddings: {input_result['similarity']:.4f}")
    print(f"Output embedding: Are the embeddings exactly the same? {output_result['are_equal']}")
    print(f"Output embedding: Cosine similarity between embeddings: {output_result['similarity']:.4f}")
