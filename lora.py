import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.merged = False

        self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.empty(out_features, rank), requires_grad=True)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.merged:
            return nn.functional.linear(x, self.weight, self.bias)
          
        base_output = nn.functional.linear(x, self.weight, self.bias)
        
        lora_down = nn.functional.linear(x, self.lora_A)
        lora_up = nn.functional.linear(lora_down, self.lora_B)
        
        return base_output + (lora_up * self.scaling)

    def merge_weights(self):
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta_w
            self.merged = True
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

def inject_lora(module, rank=8, alpha=16):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in ['query', 'value']:
            lora_layer = LoRALinear(
                in_features=child.in_features, 
                out_features=child.out_features, 
                rank=rank, 
                alpha=alpha
            )
            
            lora_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                lora_layer.bias.data = child.bias.data.clone()
            
            setattr(module, name, lora_layer)
            
        else:
            inject_lora(child, rank, alpha)

def prepare_for_lora_training(model):
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False