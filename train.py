import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr
from tqdm import tqdm

def calculate_metric(preds, labels, task_name):
    if task_name == 'mrpc':
        return {'accuracy': {accuracy_score(labels, preds)}}
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
        batch = {k: v.to_device() for k, v in batch.items()}
