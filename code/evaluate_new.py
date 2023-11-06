import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, padding=True, truncation=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_8bit=True)

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)  # Move the model to the GPU if available

    # Define the instruction template
    instruction_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{problem}\n\n"
        "### Response: Let's think step by step."
    )

    # Read the dataset
    with open(args.val_dataset_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = lines[:args.max_samples]

    pro_id_list = []

    try:
        with open(args.answers_path, 'r', encoding='utf-8-sig') as output_file:
            data = output_file.readlines()
            for line_number, line in enumerate(data, start=1):
                pro_id_list.append(json.loads(line)['queId'])
    except FileNotFoundError:
        print("No existing answer file found. Starting from scratch.")
    except Exception as e:
        print(f"An error occurred while reading the answer file: {str(e)}")

    # Process the questions in batches and generate answers
    answers_list = []
    with open(args.answers_path, 'a', encoding='utf-8') as json_file:
        for i in tqdm(range(0, len(lines), args.batch_size)):
            batch = lines[i:i + args.batch_size]
            formatted_problems = [
                instruction_template.format(
                    problem=json.loads(line)['problem'])
                for line in batch
            ]

            # Check if queId already exists in the list of generated answers
            queIds = [json.loads(line)['queId'] for line in batch]
            queIds_to_process = [q for q in queIds if q not in pro_id_list]

            # If all queIds are already generated, skip to the next batch of questions
            if not queIds_to_process:
                continue

            # Tokenize the problems
            inputs = tokenizer(formatted_problems, return_tensors='pt', padding=True, truncation=True,
                               max_length=args.max_seq_len)
            # Move the inputs to the GPU if available
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            # Generate answers using the model
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_length=args.max_seq_len, num_beams=2, temperature=1.0)

            # Decode the generated answers
            batch_answers = tokenizer.batch_decode(
                outputs, skip_special_tokens=True)

            # Extend the answers_list with the new answers
            for queId, formatted_problem, generated_text in zip(queIds, formatted_problems, batch_answers):
                if queId in pro_id_list:
                    continue

                answers_list.append({
                    'queId': queId,
                    'problem': formatted_problem,
                    'answer': generated_text
                })

                # Save the generated answer to the JSON file
                json.dump(answers_list[-1], json_file, ensure_ascii=False)
                json_file.write('\n')

                # Add the queId to the list of generated answers
                pro_id_list.append(queId)

    print(f"The answers have been saved to {args.answers_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate mathematical problems using the MetaMath model.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing questions.')
    parser.add_argument('--max_seq_len', type=int,
                        default=512, help='Maximum sequence length.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process.')
    parser.add_argument('--model_path', type=str,
                        default='../meta-math/MetaMath-7B-V1.0', help='Path to the model.')
    parser.add_argument('--tokenizer_path', type=str,
                        default='../meta-math/MetaMath-7B-V1.0', help='Path to the tokenizer.')
    parser.add_argument('--val_dataset_path', type=str,
                        default='../data/AAAI/TAL-SAQ6K-EN.jsonl', help='Path to the validation dataset.')
    parser.add_argument('--answers_path', type=str,
                        default='../data/answers.jsonl', help='Path to save the answers.')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred:{str(e)}")
