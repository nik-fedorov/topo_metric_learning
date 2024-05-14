import os

import torch


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def transform_metrics_for_wandb_logging(metrics_value):
    res = {}
    for metric_name in metrics_value:
        for k in metrics_value[metric_name]:
            res[metric_name + '/' + str(k)] = metrics_value[metric_name][k].item()
    return res


def save_model(path, num_epochs, best_metric_value, model, optimizer, scheduler=None):
    '''Save on GPU'''
    data = {
        'num_epochs': num_epochs,
        'best_metric_value': best_metric_value,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(data, path)


def load_model(path, device, model, optimizer=None, scheduler=None):
    '''Load on GPU'''
    data = torch.load(path)
    model.load_state_dict(data['model_state_dict'])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(data['scheduler_state_dict'])
    return data['num_epochs'], data.get('best_metric_value')


def exclude_from_code_logging(path, root):
    relpath = os.path.relpath(path, root)

    return (
        relpath.startswith("__pycache__/") or
        relpath.startswith(".") or
        relpath.startswith("wandb/") or
        relpath.startswith("outputs/") or
        relpath.startswith("ckpt/") or
        path.endswith('.pt') or path.endswith('.pth') or
        (
            relpath.startswith("datasets/") and
            not (path.endswith('.py') or path.endswith('.sh'))
        )
    )
