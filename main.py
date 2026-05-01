import torch
from transformers import RobertaForSequenceClassification
from data import get_dataloaders
from lora import inject_lora, prepare_for_lora_training
from train import run_training_loop

def get_fresh_model(num_labels):
    """Utility to ensure we start with a clean pre-trained model for each run."""
    return RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

def main():
    # --- Configuration ---
    # NOTE: Set epochs higher (e.g., 3-5) when running your final job on a cluster. 
    # Keep it at 1-2 for local testing to save time.
    epochs = 5
    batch_size = 16
    
    # LoRA typically requires a higher learning rate (e.g., 2e-4) than full fine-tuning (e.g., 2e-5)
    lr_full = 2e-5
    lr_lora = 2e-4
    
    # Task metadata: (dataset_name, num_labels, target_metric)
    tasks = [
        ("mrpc", 2, "accuracy"),
        ("cola", 2, "matthews_correlation"),
        ("stsb", 1, "pearson") # STS-B is regression, so it has 1 continuous output label
    ]
    
    results = {}

    print("=== Starting LoRA vs Full Fine-Tuning Evaluation Pipeline ===")

    for task_name, num_labels, metric_name in tasks:
        print(f"\n" + "="*40)
        print(f"TASK: {task_name.upper()}")
        print("="*40)
        
        # 1. Load Data
        train_loader, eval_loader, _ = get_dataloaders(task_name, batch_size=batch_size)
        results[task_name] = {}

        # 2. Full Fine-Tuning Baseline
        print(f"\n--- Running Full Fine-Tuning ({task_name.upper()}) ---")
        base_model = get_fresh_model(num_labels)
        base_metrics = run_training_loop(base_model, train_loader, eval_loader, task_name, epochs, lr_full)
        results[task_name]["Full FT"] = base_metrics[metric_name]

        # 3. LoRA Fine-Tuning
        print(f"\n--- Running LoRA Fine-Tuning ({task_name.upper()}) ---")
        lora_model = get_fresh_model(num_labels)
        inject_lora(lora_model, rank=8, alpha=16)
        prepare_for_lora_training(lora_model)
        
        lora_metrics = run_training_loop(lora_model, train_loader, eval_loader, task_name, epochs, lr_lora)
        results[task_name]["LoRA"] = lora_metrics[metric_name]

    # --- Generate the Final Table ---
    print("\n\n" + "*"*50)
    print("FINAL RESULTS TABLE")
    print("*"*50)
    print(f"{'Dataset':<10} | {'Metric':<20} | {'Full FT':<10} | {'LoRA':<10}")
    print("-" * 57)
    
    for task_name, num_labels, metric_name in tasks:
        full_score = results[task_name]['Full FT']
        lora_score = results[task_name]['LoRA']
        print(f"{task_name.upper():<10} | {metric_name:<20} | {full_score:.4f}     | {lora_score:.4f}")

if __name__ == "__main__":
    main()