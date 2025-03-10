import chardet
import os
import torch
from transformers import TrainerCallback

def detect_encoding(path):
	with open(path, "rb") as f:
		return chardet.detect(f.read(1000))["encoding"]

def determine_functions(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

def initialise_repository(root_path):
	os.makedirs(os.path.join(root_path, "models"), exist_ok=True)
	os.makedirs(os.path.join(root_path, "inputs"), exist_ok=True)

# def in_vocabulary(word):
# 	return not(MODEL_PROCESSOR.tokenizer.get_vocab().get(word) is None)

class LossLoggerCallback(TrainerCallback):
	def __init__(self, log_path):
		self.log_path = log_path
	def on_log(self, args, state, control, logs=None, **kwargs):
		if not logs: return
		loss_info = f"Step {state.global_step}: "
		if "loss" in logs:
			loss_info += f"Training Loss = {logs['loss']} "
		if "eval_loss" in logs:
			loss_info += f"Validation Loss = {logs['eval_loss']} "
		if "loss" in logs or "eval_loss" in logs:
			with open(self.log_path, "a") as f:
				f.write(loss_info.strip() + "\n")