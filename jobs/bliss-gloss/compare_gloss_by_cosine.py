# python compare_gloss_by_cosine.py data/bliss_gloss_cleaned_synonyms.json data/consine_mismatch.json

import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utility_funcs import compare_embedding, preprocess_gloss

# Set a standard deviation threshold for detecting outliers
THRESHOLD = 0.5

if len(sys.argv) != 3:
    print("Usage: python compare_gloss_by_cosine.py <input_gloss_json> <analysis_output_file>")
    sys.exit(1)

input_gloss_json = sys.argv[1]
analysis_output_file = sys.argv[2]

# Load the JSON file
with open(input_gloss_json, "r") as f:
    input_gloss_data = json.load(f)

# Load the local Llama model
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

all_single_token_glosses = {}

bliss_ids_has_input_outliers = []
bliss_ids_has_output_outliers = []
count_mismatch = 0
count_total_comparison = 0

# Process each record in the JSON
for bliss_id, glosses in input_gloss_data.items():
    single_token_glosses = []
    input_embeddings = []
    output_embeddings = []

    # Loop through each gloss to find its token id and input/output embedding
    for gloss in glosses:
        gloss = preprocess_gloss(gloss)
        tokens = tokenizer.tokenize(gloss)
        token_id = tokenizer.convert_tokens_to_ids(tokens)

        # Check if it"s a single token gloss
        if len(token_id) != 1:
            continue

        token_id = token_id[0]  # Get the single token ID
        single_token_glosses.append(gloss)
        input_embeddings.append(model.get_input_embeddings().weight[token_id])
        output_embeddings.append(model.lm_head.weight[token_id])

    num_of_single_glosses = len(single_token_glosses)
    if num_of_single_glosses > 1:
        all_single_token_glosses[bliss_id] = {
            "single_token_glosses": single_token_glosses,
            "mismatch": []
        }

        for i in range(num_of_single_glosses):
            for j in range(i + 1, num_of_single_glosses):
                count_total_comparison = count_total_comparison + 1
                input_comparison = compare_embedding(input_embeddings[i], input_embeddings[j])
                output_comparison = compare_embedding(output_embeddings[i], output_embeddings[j])

                if input_comparison["similarity"] < THRESHOLD or output_comparison["similarity"] < THRESHOLD:
                    count_mismatch = count_mismatch + 1
                    one_mismatch = {
                        "glosses": [single_token_glosses[i], single_token_glosses[j]]
                    }

                    if input_comparison["similarity"] < THRESHOLD:
                        one_mismatch["input_similarity"] = input_comparison["similarity"]
                    if output_comparison["similarity"] < THRESHOLD:
                        one_mismatch["output_similarity"] = output_comparison["similarity"]

                    all_single_token_glosses[bliss_id]["mismatch"].append(one_mismatch)

if len(all_single_token_glosses) > 0:
    print(f"{count_total_comparison} pairs compared. {count_mismatch} found with similarity lower than {THRESHOLD}")
    # Save analysis results to file
    with open(analysis_output_file, "w") as f:
        json.dump(all_single_token_glosses, f, indent=2)
