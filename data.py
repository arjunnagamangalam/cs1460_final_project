import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

task_to_keys = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

def get_dataloaders(task_name, model_name="roberta-base", batch_size=16):
    """
    Downloads the dataset, tokenizes the text, and returns PyTorch DataLoaders.
    """
    print(f"Loading {task_name.upper()} dataset...")
    
    # 1. Load the raw GLUE dataset
    raw_datasets = load_dataset("glue", task_name)
    
    # 2. Load the standard RoBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. Define the tokenization function
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    def tokenize_function(examples):
        if sentence2_key is None:
            # For CoLA (single sentence)
            return tokenizer(examples[sentence1_key], truncation=True)
        else:
            # For MRPC and STS-B (sentence pairs)
            return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    # 4. Apply tokenization to all splits (train, validation, test)
    # Using batched=True speeds this up significantly
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    
    # 5. Format the datasets for PyTorch
    # We remove the raw text columns because the model only expects input_ids and attention_masks
    columns_to_remove = [sentence1_key, "idx"]
    if sentence2_key is not None:
        columns_to_remove.append(sentence2_key)
        
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    # 6. Create the DataLoaders
    # DataCollatorWithPadding dynamically pads the batches to the longest sequence in the batch
    # This is much more efficient than padding the entire dataset to the absolute max length
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    
    # Use validation_matched or validation depending on the GLUE task
    eval_split = "validation" 
    eval_dataloader = DataLoader(
        tokenized_datasets[eval_split], 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    
    return train_dataloader, eval_dataloader, tokenizer