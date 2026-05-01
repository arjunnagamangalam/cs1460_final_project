import torch
from transformers import RobertaForSequenceClassification
from data import get_dataloaders
from lora import inject_lora, prepare_for_lora_training
from train import run_training_loop

def main():
    # 1. Configuration for the test run
    task_name = "mrpc"
    model_name = "roberta-base"
    batch_size = 16
    epochs = 1  # Just one epoch for the quick test!
    
    # LoRA typically requires a slightly higher learning rate than full fine-tuning
    lr = 2e-4   

    print(f"--- Starting End-to-End Test on {task_name.upper()} ---")

    # 2. Get Data
    train_dataloader, eval_dataloader, tokenizer = get_dataloaders(
        task_name=task_name, 
        model_name=model_name, 
        batch_size=batch_size
    )

    # 3. Load Base Model 
    # MRPC is semantic equivalence (binary classification: 0 or 1), so num_labels=2
    print("\nLoading Base RoBERTa model...")
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. Apply LoRA Surgery
    print("Injecting LoRA layers and freezing base weights...")
    inject_lora(model, rank=8, alpha=16)
    prepare_for_lora_training(model)

    # 5. Run the Optimization Loop
    print("\nStarting Training...")
    eval_metrics = run_training_loop(
        model=model, 
        train_dataloader=train_dataloader, 
        eval_dataloader=eval_dataloader, 
        task_name=task_name, 
        epochs=epochs, 
        lr=lr
    )

    # 6. Final Results
    print("\n--- Test Run Complete! ---")
    print("Final Evaluation Metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()