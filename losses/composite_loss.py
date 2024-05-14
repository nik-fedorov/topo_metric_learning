from torch import nn


class CompositeLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses, self.weights = nn.ModuleList(), []
        for loss in losses:
            if loss['weight'] != 0.0:
                self.losses.append(loss['module'])
                self.weights.append(loss['weight'])
            else:
                print(f'INFO: criterion {loss["module"]} is skipped')
        assert len(self.losses) > 0, 'All criterions in CompositeLoss have zero weights'

    def forward(self, *args, **kwargs):
        loss = 0.0
        for weight, criterion in zip(self.weights, self.losses):
            loss += weight * criterion(*args, **kwargs)
        return loss

    def summary(self):
        summary_data = dict()
        for criterion in self.losses:
            if hasattr(criterion, 'summary'):
                summary_data.update(criterion.summary())
        return summary_data
