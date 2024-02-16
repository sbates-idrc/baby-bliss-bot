from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_dir = "/home/cindyli/llama2/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

# 1. Test prompt for text generation
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=400)
pipe_return = pipe(f"<s>[INST] {prompt} [/INST]")

# write results into the result file
file_path = "/home/cindyli/llama2/original_use/result.txt"
file = open(file_path, "w")
file.write(f"1. Text generation: \n{pipe_return[0]['generated_text']}\n\n")

# 2. Test prompt for word predictions
prompt = "David is feeling sick and in the hospital. He wants to"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
predictions = pipe(prompt, max_length=20, num_return_sequences=3)

# write results into the result file
file.write("2. Word prediction: \n")
for prediction in predictions:
    file.write(f"- {prediction['generated_text']}: {prediction['score']}\n")
file.write("\n")

# 3. Test prompt for inference
prompt = "David wants to"
inference = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_length=300)[0])

# write results into the result file
file.write(f"3. Inference: \n{inference}\n")
