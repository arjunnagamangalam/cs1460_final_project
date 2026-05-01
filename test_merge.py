import torch
import time
from transformers import RobertaForSequenceClassification
from lora import inject_lora, LoRALinear

def merge_all_lora_weights(model):
    """
    Iterates through the model and triggers the merge_weights() 
    method on any CustomLoRALinear layers it finds.
    """
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged_count += 1
    print(f"Successfully merged weights in {merged_count} LoRA layers.")

def main():
    print("--- Setting up Inference Test ---")
    
    # 1. Load model and apply LoRA
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    inject_lora(model, rank=8, alpha=16)
    
    # Put model in eval mode (crucial for inference)
    model.eval()
    
    # 2. Create dummy input data (batch size 1, sequence length 128)
    # This simulates a single sentence being passed into the model
    dummy_input = torch.randint(0, 1000, (1, 128))
    dummy_mask = torch.ones((1, 128))
    
    # Disable gradient calculation since we are only testing inference
    with torch.no_grad():
        print("\n1. Running UNMERGED forward pass (Standard LoRA)...")
        start_time = time.perf_counter()
        # We run it a few times to warm up the CPU/GPU cache for a fair speed test
        for _ in range(10): 
            unmerged_outputs = model(input_ids=dummy_input, attention_mask=dummy_mask)
        unmerged_time = (time.perf_counter() - start_time) / 10
        unmerged_logits = unmerged_outputs.logits

        print("\n2. Folding matrices together (Merging Weights)...")
        merge_all_lora_weights(model)

        print("\n3. Running MERGED forward pass (Zero-Latency Inference)...")
        start_time = time.perf_counter()
        for _ in range(10):
            merged_outputs = model(input_ids=dummy_input, attention_mask=dummy_mask)
        merged_time = (time.perf_counter() - start_time) / 10
        merged_logits = merged_outputs.logits

    # 4. The Moment of Truth: Verification
    print("\n" + "="*40)
    print("INFERENCE TEST RESULTS")
    print("="*40)
    
    # We use allclose because floating point math on computers will have microscopic differences
    # at the 7th or 8th decimal place when adding matrices vs multiplying them separately.
    is_identical = torch.allclose(unmerged_logits, merged_logits, atol=1e-5)
    
    print(f"Mathematical Equivalence: {'PASSED ✅' if is_identical else 'FAILED ❌'}")
    if not is_identical:
        print(f"Unmerged Logits: {unmerged_logits}")
        print(f"Merged Logits:   {merged_logits}")
        
    print(f"\nUnmerged Latency: {unmerged_time*1000:.2f} ms per pass")
    print(f"Merged Latency:   {merged_time*1000:.2f} ms per pass")
    
    speedup = ((unmerged_time - merged_time) / unmerged_time) * 100
    if speedup > 0:
        print(f"Speed Improvement: {speedup:.1f}% faster")

if __name__ == "__main__":
    main()