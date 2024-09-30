# ### Script Summary:
# The script `clean_bliss_gloss.py` processes an input file containing Blissymbolics symbol IDs
# and their associated glosses. It performs the following tasks:
# 1. Processes the glosses to clean and standardize the format by removing certain substrings
# (i.e., anything after `_(")`) and replacing underscores (`_`) with spaces.
# 3. Outputs the cleaned data in two formats:
#    - A JSON file that contains a mapping of symbol IDs to lists of glosses.
#    - A text file that contains a sorted list of all unique glosses.
#
# Input File Structure (`<input_gloss_file>`):
# The input file is a tab-separated text file where each line represents a Bliss symbol and its associated glosses.
# The format of each line is:
# ```
# <symbol_id>    <gloss1>,<gloss2>,<gloss3>,...
# ```
# - `<symbol_id>`: A unique identifier for the Bliss symbol.
# - `<gloss1>,<gloss2>,<gloss3>,...`: A comma-separated list of glosses associated with the Bliss symbol.
#
# Output JSON File Structure (`<output_json_file>`):
# The output JSON file is a dictionary where each key is a Bliss symbol ID, and the value is a list of cleaned glosses.
# Example:
# ```json
# {
#   "00001": ["gloss1", "gloss2"],
#   "00002": ["gloss3"]
# }
# ```
#
# Output Gloss File Structure (`<output_gloss_file>`):
# The output gloss file is a plain text file containing a sorted, comma-separated list of all unique glosses across all symbols.
#
# Example:
# ```
# gloss1, gloss2, gloss3
# ```
#
# ### Script Usage:
# To run the script, you need to provide three arguments:
# 1. `<input_gloss_file>`: Path to the input file containing symbol ID and gloss mappings.
# 2. `<output_json_file>`: Path to the JSON file where the cleaned Bliss symbol mappings will be stored.
# 3. `<output_gloss_file>`: Path to the text file where the sorted list of unique glosses will be saved.
#
# Example Command:
# ```bash
# python clean_bliss_gloss.py data/bliss_gloss.txt data/cleaned_glosses.json data/unique_glosses.txt
# ```

import json
import sys

if len(sys.argv) != 4:
    print("Usage: python clean_bliss_gloss.py <input_gloss_file> <output_json_file> <output_gloss_file>")
    sys.exit(1)

input_gloss_file = sys.argv[1]
output_json_file = sys.argv[2]
output_gloss_file = sys.argv[3]


def process_gloss(gloss):
    if "_(" in gloss:
        gloss = gloss.split("_(")[0]
    return gloss.replace("_", " ")


id_gloss_dict = {}
all_glosses = set()

with open(input_gloss_file, 'r') as f:
    for line in f:
        id, glosses = line.strip().split('\t')
        processed_glosses = [process_gloss(gloss) for gloss in glosses.split(',')]
        id_gloss_dict[id] = processed_glosses
        all_glosses.update(processed_glosses)

# Write JSON output
with open(output_json_file, 'w') as f:
    json.dump(id_gloss_dict, f, indent=2)

# Write text output
with open(output_gloss_file, 'w') as f:
    f.write(', '.join(sorted(all_glosses)))
