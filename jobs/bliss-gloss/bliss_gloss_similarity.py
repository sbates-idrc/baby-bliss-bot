import os
import sys
import torch
from transformers import LlamaTokenizer, LlamaModel


def process_gloss(gloss):
    return gloss.replace('_', ' ').strip('"')


def calculate_similarity(model, tokenizer, gloss1, gloss2):
    inputs1 = tokenizer(gloss1, return_tensors="pt")
    inputs2 = tokenizer(gloss2, return_tensors="pt")

    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    return similarity.item()


if len(sys.argv) != 3:
    print("Usage: python bliss_gloss_similarity.py <file_path> <threshold>")
    sys.exit(1)

file_path = sys.argv[1]
threshold = float(sys.argv[2])

# The local directory with the model and the tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-2-7b-hf"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-2-7b-hf"

# Initialize the Llama tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaModel.from_pretrained(model_dir)

with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            print(f"*** Error: incorrect line - '{line}'")
            continue

        id, english_gloss = parts

        if not english_gloss.startswith('"'):
            continue

        glosses = [process_gloss(g) for g in english_gloss.split(',')]

        low_similarity_pairs = []
        for i in range(len(glosses)):
            for j in range(i+1, len(glosses)):
                similarity = calculate_similarity(model, tokenizer, glosses[i], glosses[j])
                if similarity < threshold:
                    low_similarity_pairs.append((glosses[i], glosses[j], similarity))

        if low_similarity_pairs:
            print(f"ID: {id}")
            print(f"Glosses: {', '.join(glosses)}")
            print("Low similarity pairs:")
            for pair in low_similarity_pairs:
                print(f"  '{pair[0]}' and '{pair[1]}': {pair[2]:.4f}")
            print()
