import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set a standard deviation threshold for detecting outliers
STD_THRESHOLD = 1

if len(sys.argv) != 3:
    print("Usage: python compare_gloss_by_std.py <input_gloss_json> <analysis_output_file>")
    sys.exit(1)

input_gloss_json = sys.argv[1]
analysis_output_file = sys.argv[2]

# Load the JSON file
with open(input_gloss_json, 'r') as f:
    input_gloss_data = json.load(f)

# Load the local Llama model
# model_dir = os.path.expanduser("~") + "/Development/LLMs/Meta-Llama-3.1-8B"
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)


def preprocess_gloss(gloss):
    return f" {gloss.replace(' ', '')}"


def get_gloss_outliers(embeddings, threshold):
    # Stack embeddings to form a 2D tensor (n_words x embedding_size)
    embeddings_tensor = torch.stack(embeddings)

    std_embedding, mean_embedding = torch.std_mean(embeddings_tensor, dim=0);

    # Calculate z-scores (how many standard deviations each word embedding is from the mean)
    z_scores = (embeddings_tensor - mean_embedding) / std_embedding

    # Compute the L2 norm of z-scores for each word (to get a single scalar value for deviation)
    l2_norm_z_scores = torch.norm(z_scores, dim=1)

    # Identify outliers based on threshold
    outlier_indices = (l2_norm_z_scores > threshold).nonzero(as_tuple=True)[0]

    # Synonyms are the embeddings that are not outliers
    synonym_indices = (l2_norm_z_scores <= threshold).nonzero(as_tuple=True)[0]
    print(f"l2_norm_z_scores: {l2_norm_z_scores}; outlier_indices: {outlier_indices}; synonym_indices: {synonym_indices}")

    return outlier_indices, synonym_indices


all_single_token_glosses = {}

bliss_ids_has_input_outliers = []
bliss_ids_has_output_outliers = []

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

        # Check if it's a single token gloss
        if len(token_id) > 1:
            continue

        # Get input and output embedding of the token
        input_embedding = model.get_input_embeddings().weight[token_id]  # the input embedding
        output_embedding = model.lm_head.weight[token_id]  # the output embedding

        single_token_glosses.append(gloss)
        input_embeddings.append(input_embedding)
        output_embeddings.append(output_embedding)

    if len(single_token_glosses) > 1:
        all_single_token_glosses[bliss_id] = {
            "single_token_glosses": single_token_glosses
        }

        input_embedding_outliers_indices, input_embedding_synonyms_indices = get_gloss_outliers(input_embeddings, STD_THRESHOLD)
        input_embedding_outlier_glosses = [glosses[i] for i in input_embedding_outliers_indices]
        print(f"input_embedding_outlier_glosses: {input_embedding_outlier_glosses}")

        output_embedding_outliers_indices, output_embedding_synonyms_indices = get_gloss_outliers(output_embeddings, STD_THRESHOLD)
        output_embedding_outlier_glosses = [glosses[i] for i in output_embedding_outliers_indices]
        print(f"output_embedding_outlier_glosses: {output_embedding_outlier_glosses}")

        shared_synonyms_indices = list(set(input_embedding_synonyms_indices) & set(output_embedding_synonyms_indices))
        synonyms = [glosses[i] for i in shared_synonyms_indices]

        if (len(synonyms) == 0):
            print(f"Error: Bliss ID {bliss_id} doesn't have synonyms within its single token glosses {glosses}")

        if len(input_embedding_outlier_glosses) > 0:
            bliss_ids_has_input_outliers.append(bliss_id)
            
        if len(output_embedding_outlier_glosses) > 0:
            bliss_ids_has_output_outliers.append(bliss_id)
            
        all_single_token_glosses[bliss_id] = {
            "synonum_glosses": synonyms
        }

        if len(input_embedding_outlier_glosses) > 0:
            all_single_token_glosses[bliss_id] = {
                "input_embedding_outlier_glosses": input_embedding_outlier_glosses
            }
        if len(output_embedding_outlier_glosses) > 0:
            all_single_token_glosses[bliss_id] = {
                "output_embedding_outlier_glosses": output_embedding_outlier_glosses
            }


if len(bliss_ids_has_input_outliers) > 0:
    print(f"{len(bliss_ids_has_input_outliers)} Bliss symbols have outliers in input embedding")
if len(bliss_ids_has_output_outliers) > 0:
    print(f"{len(bliss_ids_has_output_outliers)} Bliss symbols have outliers in output embedding")

# Save added tokens to file
with open(analysis_output_file, 'w') as f:
    json.dump(all_single_token_glosses, f, indent=2)
