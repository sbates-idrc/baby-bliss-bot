import os
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


# The local directory with the model and the tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-2-7b-hf"

# Initialize the Llama tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaModel.from_pretrained(model_dir)

line = '8486	"period,point,full_stop,decimal_point"'
parts = line.strip().split('\t')
if len(parts) != 2:
    print(f"*** Error: incorrect line - '{line}'")
    exit()

id, english_gloss = parts

if not english_gloss.startswith('"'):
    print(f"*** Error: Gloss not double quoted - '{line}'")
    exit()

glosses = [process_gloss(g) for g in english_gloss.split(',')]

for i in range(len(glosses)):
    for j in range(i+1, len(glosses)):
        similarity = calculate_similarity(model, tokenizer, glosses[i], glosses[j])
        print(f"ID: {id}")
        print(f"Glosses: {', '.join(glosses)}")
        print("Compared:")
        print(f"  '{glosses[i]}' and '{glosses[j]}': {similarity:.4f}")
        print()
