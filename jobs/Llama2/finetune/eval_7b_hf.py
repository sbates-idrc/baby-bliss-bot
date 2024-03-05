import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_dir = "/home/cindyli/llama2/finetune/results-finetune-7b-hf-4epochs/checkpoint-950/"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    model_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


# Evaluation
# Inference. Generated texts by giving temperature 0.1 to 0.9 in the step of 0.1
def generate_text(instruction, input, model, tokenizer):
    input_key = "### Input:\n"
    response_key = "### Response:\n"
    prompt = f"{instruction}{input_key}{input}\n\n{response_key}\n"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    print(f"Instruction: {instruction}\n")
    print(f"Prompt: {input}\n")
    for temperature in range(0.1, 0.9, 0.1):
        outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=temperature)
        print(f"Generated instruction (temprature {temperature}): {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}\n\n")


print("1. Inference\n")
instruction = "### Instruction: \nConvert the input English sentence to a Bliss sentence.\n\n"
input = "I am a programmer."
generate_text(instruction, input, model, tokenizer)

input = "Joe will explore the picturesque landscapes of a charming countryside village tomorrow."
generate_text(instruction, input, model, tokenizer)

input = "I had the pleasure of watching a captivating movie that thoroughly engaged my senses and emotions, providing a delightful escape into the realm of cinematic storytelling."
generate_text(instruction, input, model, tokenizer)

instruction = "### Instruction: \nConvert the input Bliss sentence to a English sentence.\n\n"
input = "past:The girl run in the park."
generate_text(instruction, input, model, tokenizer)

input = "future:month next, I embark on an journey exciting to explore the cultures vibrant and landscapes breathtaking of Southeast Asia."
generate_text(instruction, input, model, tokenizer)
