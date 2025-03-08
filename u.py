import chardet
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def detect_encoding(path):
	with open(path, "rb") as f:
		return chardet.detect(f.read(100000))["encoding"]

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
	processor: Any
	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		input_features = [{"input_features": feature["input_features"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]
		batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
		labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
		batch["labels"] = labels
		return batch