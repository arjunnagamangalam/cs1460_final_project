import torch
from transformers import RobertaForSequenceClassification
from lora import inject_lora, prepare_for_lora_training

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}%"
    )

if __name__ == "__main__":
    print("Downloading/Loading base RoBERTa model...")
    # We load the base model with 2 labels (e.g., for a binary classification task like MRPC)
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    
    print("\n--- Base Model (Full Fine-Tuning) ---")
    # By default, all parameters in a newly loaded model require gradients
    print_trainable_parameters(model)
    
    print("\nInjecting LoRA layers and freezing base weights...")
    # Perform the architecture surgery
    inject_lora(model, rank=8, alpha=16)
    prepare_for_lora_training(model)
    
    print("\n--- LoRA Adapted Model ---")
    print_trainable_parameters(model)