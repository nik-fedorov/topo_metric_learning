import torch

from oml.functional.metrics import (
    calc_gt_mask,
    calc_mask_to_ignore,
    calc_retrieval_metrics,
)


def compute_metrics(dist_mat, labels, is_query, is_gallery, **metrics):
    mask_gt = calc_gt_mask(labels=labels, is_query=is_query, is_gallery=is_gallery)
    mask_to_ignore = calc_mask_to_ignore(is_query=is_query, is_gallery=is_gallery)
    return calc_retrieval_metrics(dist_mat, mask_gt, mask_to_ignore, **metrics)


@torch.no_grad()
def track_additional_valid_metrics(embeds, labels, dist_mat):
    additionel_metrics = {}

    class_centers, class_sizes = [], []
    for label, count in zip(*torch.unique(labels, return_counts=True)):
        class_embeds = embeds[labels == label]
        class_center = torch.mean(class_embeds, dim=0)
        class_centers += [class_center]
        class_variance = torch.sum((class_embeds - class_center.unsqueeze(0)) ** 2) / count
        class_sizes += [torch.sqrt(class_variance)]

    class_sizes = torch.tensor(class_sizes)
    class_centers = torch.stack(class_centers, dim=0)
    class_centers_dist_mat = torch.cdist(class_centers, class_centers, p=2)
    n_classes = class_centers.shape[0]

    additionel_metrics['additional/class_sizes/min'] = torch.min(class_sizes)
    additionel_metrics['additional/class_sizes/max'] = torch.max(class_sizes)
    additionel_metrics['additional/class_sizes/mean'] = torch.mean(class_sizes)

    additionel_metrics['additional/inter_class_dist/min'] = \
        torch.min(class_centers_dist_mat[class_centers_dist_mat > 0])
    additionel_metrics['additional/inter_class_dist/max'] = torch.max(class_centers_dist_mat)
    additionel_metrics['additional/inter_class_dist/mean'] = \
        torch.sum(class_centers_dist_mat) / float(n_classes) / (n_classes - 1)

    return additionel_metrics
