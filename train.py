import datetime as dt
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import oml
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.transforms.images.albumentations import (
    get_augs_albu,
    get_normalisation_resize_albu
)
from oml.utils.misc import set_global_seed

import wandb

from constants import CKPT_DIR_NAME, METRICS
import datasets
import losses
from metrics import compute_metrics, track_additional_valid_metrics
import models
from utils import (
    transform_metrics_for_wandb_logging,
    save_model,
    load_model,
    exclude_from_code_logging
)


@torch.no_grad()
def inference(model, valid_loader, device):
    embeds, labels = [], []
    for batch in tqdm(valid_loader):
        embeds += [model(batch['input_tensors'].to(device))]
        labels += [batch['labels']]
    return torch.cat(embeds, dim=0).cpu(), torch.cat(labels, dim=0).cpu()


@torch.no_grad()
def validation(model, valid_loader, metrics, device):
    model.eval()
    embeds, labels = inference(model, valid_loader, device)
    print(f'Inference finished: {dt.datetime.now()}')

    dist_mat = torch.cdist(embeds, embeds, p=2)
    mask = torch.ones(len(embeds))
    metrics_value = compute_metrics(dist_mat, labels, mask, mask, **metrics)
    wandb_metrics_value = transform_metrics_for_wandb_logging(metrics_value)
    wandb_metrics_value.update(track_additional_valid_metrics(embeds, labels, dist_mat))
    print(wandb_metrics_value, end='\n\n')

    return wandb_metrics_value


@torch.no_grad()
def add_train_metrics(wandb_metrics_value, model, train_loader, metrics, device):
    train_metrics_value = validation(model, train_loader, metrics, device)
    for metric, value in train_metrics_value.items():
        wandb_metrics_value['train/' + metric] = value


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    set_global_seed(cfg.global_seed)

    train_transforms = hydra.utils.instantiate(cfg.train_transforms)
    valid_transforms = hydra.utils.instantiate(cfg.valid_transforms)

    dataset_class = getattr(datasets, cfg.dataset.class_name)
    train_dataset = dataset_class(transforms=train_transforms, is_train=True)
    train_dataset_metrics = dataset_class(transforms=valid_transforms, is_train=True)
    valid_dataset = dataset_class(transforms=valid_transforms, is_train=False)

    sampler = BalanceSampler(train_dataset.get_labels(),
                             n_labels=cfg.dataloader.n_labels,
                             n_instances=cfg.dataloader.n_instances)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=cfg.dataloader.num_workers)
    train_loader_metrics = DataLoader(train_dataset_metrics, batch_size=cfg.dataloader.valid_batch_size, num_workers=cfg.dataloader.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.dataloader.valid_batch_size, num_workers=cfg.dataloader.num_workers)

    model = hydra.utils.instantiate(cfg.model)
    criterion = hydra.utils.instantiate(cfg.criterion)
    model.add_module('criterion', criterion)  # some criterions have trainable params, we want to save them in ckpts
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    scheduler = None

    wandb_init_data = OmegaConf.to_container(cfg.wandb_init_data)
    wandb_init_data['config'] = OmegaConf.to_container(cfg)

    # prepare dir for checkpoints
    ckpt_dir = Path(CKPT_DIR_NAME) / str(dt.datetime.now()).replace(' ', '_')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with wandb.init(**wandb_init_data) as run:
        if wandb.run.resumed:
            print('Restoring checkpoint...')
            ckpt_path = cfg.starting_checkpoint.ckpt_path
            try:
                path = wandb.restore(ckpt_path, replace=True)
                start_epoch, best_cmc1 = load_model(path, device, model, optimizer, scheduler)
                print('Wandb checkpoint successfully restored!')
            except UnicodeDecodeError as e:
                print(f'Cannot restore from wandb: {e}')
                print(f'Restoring from local storage...')
                start_epoch, best_cmc1 = load_model(ckpt_path, device, model, optimizer, scheduler)
                print('Local checkpoint successfully restored!')
                print(f'Continue training from epoch {start_epoch}, best metric {best_cmc1}')

        else:
            # log all source code
            wandb.run.log_code(
                ".",
                include_fn=lambda path: True,
                exclude_fn=exclude_from_code_logging
            )

            # log config
            # config_name = hydra.core.hydra_config.HydraConfig.get().job.config_name
            # wandb.run.log_artifact(f"configs/{config_name}", name="config")

            # run first evaluation before training
            # print('Evaluating pre-trained model before training')
            # wandb_metrics_value = validation(model, valid_loader, METRICS, device)
            # add_train_metrics(wandb_metrics_value, model, train_loader_metrics, METRICS, device)
            # wandb.log(wandb_metrics_value)
            # best_cmc1 = wandb_metrics_value['cmc/1']
            best_cmc1 = None

            start_epoch = 0

        for epoch in range(start_epoch, cfg.trainer.n_epochs):
            model.train()
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                embeddings = model(batch['input_tensors'].to(device))
                loss = criterion(
                    input_tensors=batch['input_tensors'].to(device),
                    embeddings=embeddings,
                    labels=batch['labels'].to(device),
                    epoch=epoch,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip_grad_norm)
                optimizer.step()

            if (epoch + 1) % cfg.trainer.valid_period == 0:
                print(f'{epoch + 1} training epochs finished\nValidation started: {dt.datetime.now()}')
                with torch.inference_mode():
                    wandb_metrics_value = validation(model, valid_loader, METRICS, device)
                    add_train_metrics(wandb_metrics_value, model, train_loader_metrics, METRICS, device)
                    wandb_metrics_value.update(criterion.summary())
                    wandb.log(wandb_metrics_value)

                    # saving last checkpoint
                    last_ckpt_path = str(ckpt_dir / 'last.pt')
                    save_model(last_ckpt_path, epoch + 1, best_cmc1, model, optimizer, scheduler)
                    wandb.save(last_ckpt_path)

                    # update the best checkpoint if needed
                    if best_cmc1 is None or wandb_metrics_value['cmc/1'] > best_cmc1:
                        best_cmc1 = wandb_metrics_value['cmc/1']
                        best_ckpt_path = str(ckpt_dir / 'best.pt')
                        save_model(best_ckpt_path, epoch + 1, best_cmc1, model, optimizer, scheduler)
                        wandb.save(best_ckpt_path)
                        print(f'\nNew best CMC@1 {best_cmc1} at {epoch + 1} epoch\n')


if __name__ == '__main__':
    main()
