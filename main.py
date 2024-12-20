import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_evaluate
from identifier_generifier import IdentifierGenerifier

def evaluate_model(model_name: str, task: str, language: str, device: str, generifier: IdentifierGenerifier):
	# Check task support
	if task not in TASK_CONFIGS:
		raise ValueError(f"Unsupported task: {task}. Supported tasks are: {list(TASK_CONFIGS.keys())}")

	# Load task configuration
	task_config = TASK_CONFIGS[task]

	# Load dataset
	print(f"Loading dataset for task: {task} - {language}...")
	dataset = load_dataset(task_config["dataset"], language)
	test_data = dataset[task_config["data_split"]]

	# Print dataset sizE
	print(f"Test data size: {len(test_data)}")

	# Load evaluation metric
	metric = load_evaluate(task_config["metric"])

	# Load model and tokenizer
	print(f"Loading model {model_name}...")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)
	model.to(device)

	# Evaluation loop
	print("Evaluating...")
	predictions = []
	references = []

	i = 0
	for sample in test_data:
		print(i)
		# Prepare input and reference based on task
		if task == "code-to-text":
			input_text = sample["code"]
			reference = sample["docstring"]
		elif task == "code-completion":
			input_text = sample["input"]
			reference = sample["output"]
		else:
			raise ValueError(f"Unsupported task: {task}")

		print(f"input_text:\n{input_text}")
		processed_text = generifier.generify(input_text)
		print(f"processed_text:\n{processed_text}")

		# Tokenize input
		inputs = tokenizer(processed_text, return_tensors="pt").to(device)
		# Print the number of tokens in inputs
		num_tokens = inputs["input_ids"].size(1)
		print(f"Number of tokens in input: {num_tokens}")

		# Skip processing if the number of tokens exceeds the model limit
		if num_tokens > tokenizer.model_max_length:
			print(f"Skipping sample: {num_tokens} tokens exceed model's max length of {tokenizer.model_max_length}")
			continue

		# Generate output
		with torch.no_grad():
			outputs = model.generate(**inputs, max_new_tokens=128)

		# Decode predictions
		predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
		predictions.append(predicted_text)
		references.append(reference)

		print(f"prediction:\n{predicted_text}")
		print(f"reference:\n{reference}")

		i += 1
		if i == 1:
			break

	# Compute metrics
	results = metric.compute(predictions=predictions, references=references)
	print(f"Results for task {task}: {results}")
	return results

if __name__ == "__main__":

	# Define supported CodeXGLUE tasks and their configurations
	TASK_CONFIGS = {
		"code-to-text": {
			"dataset": "code_x_glue_ct_code_to_text",
			"metric": "bleu",
			"data_split": "test",
		},
		"code-completion": {
			"dataset": "code_x_glue_tc_code_completion",
			"metric": "accuracy",
			"data_split": "test",
		},
		# Add other tasks here
	}

	parser = argparse.ArgumentParser(description="Evaluate Hugging Face LLMs on CodeXGLUE tasks.")
	parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name",
		choices=["gpt2"])
	parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIGS.keys()), help="CodeXGLUE task",
		choices=["code-to-text", "code-completion"])
	parser.add_argument("--language", type=str, required=True, help="Programming language for the task",
		choices=["python"])
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation.")

	args = parser.parse_args()

	if (os.environ["HF_HOME"] != "/vol/bitbucket/ayj20/hf"):
		print("Set HF_HOME variable")
		quit()

	generifier = IdentifierGenerifier(args.language)

	evaluate_model(args.model_name, args.task, args.language, args.device, generifier)
