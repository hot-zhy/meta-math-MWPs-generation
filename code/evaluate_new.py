"""
This script evaluates a set of mathematical problems using the MetaMath model.
It processes questions in batches, formats them according to a given template,
and generates answers which are then saved to a JSON file with each entry
containing the question ID, the original problem, and the model-generated answer.
"""

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

# Adjust batch size based on model and machine capabilities
batch_size = 8
max_seq_len = 1024
max_samples = 100

# Specify the path to your local model
model_path = "../meta-math/MetaMath-7B-V1.0"
tokenizer_path = "../meta-math/MetaMath-7B-V1.0"

# Specify the path to your dataset and where to save the answers
val_dataset_path = "../data/AAAI/TAL-SAQ6K-EN.jsonl"
answers_path = "../data/answers.json"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set up the pipeline for text generation
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Check if the model is on CUDA
print(next(text_generator.model.parameters()).is_cuda)

# Define the instruction template
instruction_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{problem}\n\n"
    "### Response: Let's think step by step."
)

# Read the dataset
with open(val_dataset_path, 'r') as file:
    lines = file.readlines()
lines = lines[:max_samples]

# Process the questions in batches and generate answers
answers_list = []
for i in range(0, len(lines), batch_size):
    batch = lines[i:i + batch_size]
    formatted_problems = [
        instruction_template.format(problem=json.loads(line)['problem'])
        for line in batch
    ]
    queIds = [json.loads(line)['queId'] for line in batch]
    
    # Generate answers using the text_generator
    batch_answers = text_generator(formatted_problems, max_length=200)

    # Extend the answers_list with the new answers
    for queId, formatted_problem, answer_dict in zip(queIds, formatted_problems, batch_answers):
        # Since each answer_dict is a list with one dictionary
        # We access the first element of the list and then the 'generated_text' key
        generated_text = answer_dict[0]['generated_text'].strip()
        answers_list.append({
            'queId': queId,
            'problem': formatted_problem,
            'answer': generated_text
        })
        
# Save the results to a JSON file
with open(answers_path, 'w') as json_file:
    json.dump(answers_list, json_file, indent=4)

print(f"The answers have been saved to {answers_path}")
