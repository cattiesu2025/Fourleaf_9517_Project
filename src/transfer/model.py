"""
src/transfer/model.py
----------------------
Builds ResNet18 for transfer learning.

Official method names (per 数据与输出格式规范 v1.1, Section 6 命名清单):
    resnet18_pretrained_frozen
    resnet18_pretrained_finetuned

Only these two are "official" deliverables. "layer4" (partial unfreeze)
is kept here as an optional extra ablation for your own exploration /
report discussion, but its outputs should NOT be submitted under the
official method-name directories -- if you want to report it, treat it
as a supplementary result and clear the naming with the group first
(per the "如需扩展先在群里同步" rule).
"""

import torch.nn as nn
import torchvision.models as models

from src.transfer.attention import MultiHeadSelfAttention2D

# Maps our internal strategy keys -> the official method_name string
STRATEGY_TO_METHOD_NAME = {
    "frozen": "resnet18_pretrained_frozen",
    "finetuned": "resnet18_pretrained_finetuned",
    # "layer4" intentionally omitted: not in the official naming list.
}


class AttentionResNet18(nn.Module):
    """ResNet18 with a multi-head self-attention block inserted after layer4
    (7x7 feature map -> 49 spatial tokens), before global average pooling.

    NOT part of the official method-name list -- this is the optional
    "attention- and part-based models" extra direction from the spec's
    Advanced Method Development section. Clear with the group before
    treating results from this as a reportable official method.
    """

    def __init__(self, base_resnet: nn.Module, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Reuse the pretrained/scratch backbone's layers directly (no re-init).
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool
        self.fc = base_resnet.fc

        # layer4's output channel count for standard ResNet18 is 512.
        attn_channels = self.layer4[-1].conv2.out_channels
        self.attention = MultiHeadSelfAttention2D(attn_channels, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.attention(x)  # the only structural addition vs. plain ResNet18

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def build_model(num_classes: int, pretrained: bool = True,
                 use_attention: bool = False, num_heads: int = 8,
                 dropout_rate: float = 0.0) -> nn.Module:
    """
    Args:
        num_classes: must be 500 for official runs (per split_config.json).
        pretrained: True -> load ImageNet weights (needs internet on first use).
        use_attention: True -> insert a multi-head self-attention block after
            layer4 (extra ablation, not an official method_name -- see
            AttentionResNet18 docstring).
        num_heads: number of attention heads, only used when use_attention=True.
        dropout_rate: if > 0, inserts nn.Dropout(dropout_rate) right before
            the final fc layer -- a regularization knob for combating
            overfitting (see the frozen/finetuned train-vs-val gap analysis).
            0.3-0.5 is a reasonable starting range for a fc-only regularizer.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features

    if dropout_rate > 0:
        model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)

    if use_attention:
        model = AttentionResNet18(model, num_heads=num_heads)

    return model


def set_finetune_strategy(model: nn.Module, strategy: str) -> nn.Module:
    """
    strategy:
        "frozen"    -> freeze everything except fc (linear probe)
        "finetuned" -> unfreeze everything (full fine-tuning)
        "layer4"    -> [extra, non-official] unfreeze layer4 + fc only

    If model is an AttentionResNet18, the attention block is ALWAYS kept
    trainable regardless of strategy -- it has no pretrained weights (random
    init), so freezing it would leave it useless noise mixed into the features.
    """
    if strategy not in {"frozen", "finetuned", "layer4"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    for param in model.parameters():
        param.requires_grad = False

    if strategy == "frozen":
        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "finetuned":
        for param in model.parameters():
            param.requires_grad = True

    elif strategy == "layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    if isinstance(model, AttentionResNet18):
        for param in model.attention.parameters():
            param.requires_grad = True

    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
