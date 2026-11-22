
import torch
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation
)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

print("Loading pretrained SegFormer-B0 segmentation model...")

config = SegformerConfig.from_pretrained(MODEL_NAME)

config.num_labels = 2
config.id2label = {
    0: "background",
    1: "boundary"
}
config.label2id = {
    "background": 0,
    "boundary": 1
}

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Model loaded successfully!")
print("Device:", device)
print("Number of parameters:",
      sum(p.numel() for p in model.parameters()))
print("Number of output classes:", model.config.num_labels)
