from transformers import AdamW
from torch.optim import lr_scheduler


def adam(params, config=None):
    if config is None:
        config = {
            'lr': 3e-5,
            'warmup_steps': 5,
            'weight_decay': 0.01
        }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0}
    ]

    optim = AdamW(optimizer_grouped_parameters, lr=config['lr'], correct_bias=False) 
    scheduler = lr_scheduler.LinearLR(optim, total_iters=config['warmup_steps'])  # PyTorch scheduler
    return optim, scheduler