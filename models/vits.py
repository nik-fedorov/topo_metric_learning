import timm
import torch.nn.functional as F
import torch.nn as nn

from utils import freeze


def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class ViTS(nn.Module):
    def __init__(self, num_freezed_blocks=0, freeze_patch_embeds=True,
                 normalize_output_embeddings=False):
        super().__init__()

        self.num_freezed_blocks = num_freezed_blocks
        self.freeze_patch_embeds = freeze_patch_embeds

        self.body = timm.create_model('vit_small_patch16_224', pretrained=True)
        # self.body = torch.hub.load("facebookresearch/dino:main", 'dino_vits16')
        # self.body = ViTExtractor('vits8_dino', arch='vits8', normalise_features=False)

        if freeze_patch_embeds:
            freeze(self.body.patch_embed)  # freeze MLPs for patch embeds

        for i in range(num_freezed_blocks):
            freeze(self.body.blocks[i])    # freeze first num_freezed_blocks transformer blocks in ViT

        #         self.head = nn.Sequential(nn.Linear(384, 384), NormLayer())
        #         nn.init.constant_(self.head[0].bias.data, 0)
        #         nn.init.orthogonal_(self.head[0].weight.data)
        rm_head(self.body)

        if normalize_output_embeddings:
            self.head = NormLayer()
        else:
            self.head = nn.Identity()

    def forward(self, x):
        return self.head(self.body(x))
