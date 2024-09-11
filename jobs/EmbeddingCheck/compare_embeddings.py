# Compare the embedding of the same word in difference sentences(contexts)

import os
from transformers import LlamaTokenizer, LlamaModel
import torch
import torch.nn.functional as F

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

    # Extract the embedding for the word
    word_embedding = outputs.last_hidden_state[0, word_position, :]

    return word_embedding


# Function to calculate cosine similarity between two embeddings
def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


for i, example in enumerate(examples, 1):
    print(f"\nExample {i}: Word '{example['word']}'")
    print("-" * 50)

    print(f"Phrase 1: '{example['phrases'][0]}'")
    print(f"Phrase 2: '{example['phrases'][1]}'")

    # Get embeddings for the word in both phrases
    embeddings = [get_word_embedding(phrase, example['word']) for phrase in example['phrases']]

    # Compare the embeddings
    are_equal = torch.allclose(embeddings[0], embeddings[1], atol=1e-6)
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    distance = 1 - similarity

    # Print results
    print(f"Are the embeddings exactly the same? {are_equal}")
    print(f"Cosine similarity between embeddings: {similarity:.4f}")
    print(f"Cosine distance between embeddings: {distance:.4f}")
