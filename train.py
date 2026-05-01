import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr
from tqdm import tqdm

def calculate_metric(preds, labels, task_name):
    if task_name == 'mrpc':
        return {'accuracy': accuracy_score(labels, preds)}
    elif task_name == 'cola':
        return {'matthews_correlation': matthews_corrcoef(labels, preds)}
    elif task_name == 'stsb':
        corr, _ = pearsonr(preds, labels)
        return {'pearson': corr}

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)

def evaluate(model, dataloader, task_name, device):
    """
    Runs evaluation on the validation set and calculates the required metric.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            # STS-B is regression (1 float output), MRPC/CoLA are classification (logits)
            if task_name == "stsb":
                preds = outputs.logits.squeeze().cpu().numpy()
            else:
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                
            labels = batch["labels"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metric(all_preds, all_labels, task_name)
    metrics["eval_loss"] = avg_loss
    
    return metrics

def run_training_loop(model, train_dataloader, eval_dataloader, task_name, epochs=3, lr=2e-5):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    print(f"Device: {device}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        eval_metrics = evaluate(model, eval_dataloader, task_name, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        for k, v in eval_metrics.items():
            print(f"Eval {k.capitalize()}: {v:.4f}")
            
    return eval_metrics