import einops
import timm
import timm.optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch import Tensor


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_multiplier: int = 1,
        pre_ln: bool = False,
    ) -> None:
        super().__init__()
        self.pre_ln = pre_ln
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_multiplier),
            nn.ReLU(),
            nn.Linear(dim * ffn_multiplier, dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_ln = nn.LayerNorm(dim)

    def forward(self, q: Tensor, kv: Tensor | None = None, need_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        if self.pre_ln:
            q = self.mha_ln(q)
            kv = q if kv is None else self.mha_ln(kv)
            x, attn = self.mha(q, kv, kv, need_weights=need_weights)
            x = q + x
            x = self.ffn_ln(x)
            x = x + self.ffn_dropout(self.ffn(x))
        else:
            kv = q if kv is None else kv
            x, attn = self.mha(q, kv, kv, need_weights=need_weights)
            x = self.mha_ln(q + x)
            x = self.ffn(x)
            x = self.ffn_ln(x + self.ffn_dropout(x))

        return x, attn


class PixelRelationEncoder(nn.Module):
    def __init__(
        self,
        features_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: int = 1,
        pre_ln: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(features_dim, embed_dim)
        self.transformer = TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            pre_ln=pre_ln,
        )

    def forward(self, feature_maps: Tensor) -> Tensor:
        x = einops.rearrange(feature_maps, "B D H W -> B (H W) D")  # (B, HW, D)
        x = self.proj(x)  # (B, HW, L)
        x, _ = self.transformer(x)  # (B, HW, L)
        return x


class LesionFilterDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_filters: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: int = 1,
        pre_ln: bool = False,
    ) -> None:
        super().__init__()
        self.filters = nn.Parameter(torch.randn(num_filters, dim))
        self.filters_transformer = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            pre_ln=pre_ln,
        )

        self.cross_transformer = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            pre_ln=pre_ln,
        )

    def forward(self, features: Tensor, need_maps: bool = False) -> tuple[Tensor, Tensor | None]:
        filters = self.filters.expand(features.shape[0], -1, -1)  # (B, K, L)
        filters, _ = self.filters_transformer(filters)  # (B, K, L)

        x, m = self.cross_transformer(filters, features, need_weights=need_maps)  # (B, K, L), (B, K, HW)
        return x, m


class LesionAwareTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        num_filters: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: int = 1,
        pre_ln: bool = False,
        triplet_margin: float = 1.0,  # TODO: figure out a good default value
        w_triplet: float = 0.04,
        w_consistency: float = 0.01,
        optimizer: str = "adamw",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.triplet_margin = triplet_margin
        self.w_consistency = w_consistency
        self.w_triplet = w_triplet

        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(-1,))
        self.pixel_relation_encoder = PixelRelationEncoder(
            features_dim=self.backbone.feature_info.channels(-1),
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            pre_ln=pre_ln,
        )
        self.lesion_filter_decoder = LesionFilterDecoder(
            dim=embed_dim,
            num_filters=num_filters,
            num_heads=num_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            pre_ln=pre_ln,
        )

        self.filter_importance = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(embed_dim * num_filters, num_classes * num_filters)

        self.register_buffer("class_centers", torch.randn(num_classes, embed_dim))
        self.register_buffer("class_tally", torch.zeros(num_classes))

    def forward(self, img: Tensor, need_maps: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        (feature_maps,) = self.backbone(img)  # (B, D, H, W)
        f = self.pixel_relation_encoder(feature_maps)  # (B, HW, L)
        x, m = self.lesion_filter_decoder(f, need_maps)  # (B, K, L), (B, K, HW)

        t = self.filter_importance(self.lesion_filter_decoder.filters)  # (K, 1)
        t = t.squeeze(-1)  # (K,)

        if m is not None:
            m = einops.rearrange(m, "B K (H W) -> B H W K", H=feature_maps.shape[-2])
            a = m @ t  # (B, H, W)
        else:
            a = None

        x = einops.rearrange(x, "B K L -> B (K L)")  # (B, K * L)
        y = self.classifier(x)  # (B, K * C)
        y = einops.rearrange(y, "B (K C) -> B K C", K=t.shape[0])  # (B, K, C)
        logits = torch.einsum("bkc,k->bc", y, t)  # (B, C)

        return logits if not need_maps else (logits, a)

    def training_step(self, image: Tensor, gt: Tensor) -> Tensor:
        (feature_maps,) = self.backbone(image)  # (B, D, H, W)
        f = self.pixel_relation_encoder(feature_maps)  # (B, HW, L)
        x, _ = self.lesion_filter_decoder(f)  # (B, K, L), None

        t = self.filter_importance(self.lesion_filter_decoder.filters)  # (K, 1)
        t = t.squeeze(-1)  # (K,)

        x_ = einops.rearrange(x, "B K L -> B (K L)")  # (B, K * L)
        y = self.classifier(x_)  # (B, K * C)
        y = einops.rearrange(y, "B (K C) -> B K C", K=t.shape[0], C=y.shape[1] // t.shape[0])  # (B, K, C)
        logits = torch.einsum("bkc,k->bc", y, t)  # (B, C)

        cls_loss = F.cross_entropy(logits, gt)
        triplet_loss = self.triplet_loss(x)
        consistency_loss = self.consistency_loss(x, t, gt)

        loss = cls_loss + self.w_triplet * triplet_loss + self.w_consistency * consistency_loss
        return loss

    def triplet_loss(self, x: Tensor) -> Tensor:
        B, K, L = x.shape
        x_ = F.normalize(x, dim=-1)

        pos_dot = torch.einsum("mkl,qkl->mkq", x_, x_)
        mask = torch.eye(B, dtype=torch.bool, device=x.device).unsqueeze(1).expand(B, K, B)
        pos_dot.masked_fill_(mask, float("inf"))
        pos = pos_dot.min(dim=-1).values

        neg_dot = torch.einsum("mkl,nzl->mknz", x_, x_)
        mask = torch.eye(K, dtype=torch.bool, device=x.device).unsqueeze(0).unsqueeze(2).expand(B, K, B, K)
        neg_dot.masked_fill_(mask, -float("inf"))
        neg = neg_dot.max(dim=-1).values.max(dim=-1).values

        return torch.mean((pos - neg + self.triplet_margin).clamp(min=0))

    def consistency_loss(self, x: Tensor, t: Tensor, target: Tensor) -> Tensor:
        overall = torch.einsum("bkl,k->bl", x, t)  # (B, L)
        overall = torch_scatter.scatter_mean(overall, target, dim=0, dim_size=self.num_classes)  # (C, L)
        gcl = torch.norm(overall - self.class_centers, p=2, dim=-1).mean()

        # update class centers with ema
        self.class_tally += torch_scatter.scatter(torch.ones_like(target), target, dim=0, dim_size=self.num_classes)
        eta = torch.exp(-self.class_tally).unsqueeze(-1)  # (C, 1)
        self.class_centers = (1 - eta) * self.class_centers + eta * overall.detach()

        return gcl
