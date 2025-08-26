import torch
import torchvision
import torch.nn as nn
from typing import Tuple
import math 

class PatchEmbedding(nn.Module):
    def __init__(
        self, img_size: int, in_chanels: int, hidden_dim: int, patch_size: int
    ) -> None:
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.liner_projection = nn.Conv2d(
            in_chanels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor):

        x = self.liner_projection(
            x
        )  # b, in_channels, w, h -> b, hid_dim, num_patch, num_patch
        x = x.flatten(2).transpose(
            1, 2
        )  # b, hid_dim, num_patch, num_patch -> b, num_patch*num_ptch, hid_dim
        return x


class Embedding(nn.Module):

    def __init__(
        self, img_size: int, in_chanels: int, hidden_dim: int, patch_size: int, p: float
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size, in_chanels, hidden_dim, patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.positional_embedings = nn.Parameter(
            torch.randn(1, self.patch_embedding.num_patches + 1, hidden_dim)
        )  # 1, num_patches + 1 , hidden_dim -> 1 for cls token
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)  # b, num_patches, hidden_dim
        bs = x.shape[0]
        # to concat cls token to patch embedings expand to bs, 1, hidden_dim
        cls_t = self.cls_token.expand(bs, -1, -1)
        x = torch.cat(
            (cls_t, x), dim=1
        )  # bs, 1, hidden_dim + bs, num_patch, hidden_dim
        x = x + self.positional_embedings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):

    def __init__(
        self, hidden_dim: int, attention_head_dim: int, p: float, bias: bool = True
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_head_dim = attention_head_dim
        self.q = nn.Linear(hidden_dim, attention_head_dim, bias=bias)
        self.k = nn.Linear(hidden_dim, attention_head_dim, bias=bias)
        self.v = nn.Linear(hidden_dim, attention_head_dim, bias=bias)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        q, k, v = self.q(x), self.k(x), self.v(x)
        # softmax(q*k'/sqrt(d)) * v
        attention_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.attention_head_dim
        )
        attention_prob = nn.functional.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        attention = torch.matmul(attention_prob, v)

        return attention_prob, attention


class MultiHead(nn.Module):
    def __init__(
        self, hidden_dim: int, num_attention_heads: int, p: float, biase: bool
    ) -> None:
        super().__init__()

        attention_head_dim = hidden_dim // num_attention_heads
        all_head_dim = num_attention_heads * attention_head_dim
        self.heads = nn.ModuleList([])
        for _ in range(num_attention_heads):
            head = AttentionHead(hidden_dim, attention_head_dim, p, biase)
            self.heads.append(head)

        self.output_projection = nn.Linear(all_head_dim, hidden_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor, output_attenstion: bool = False):

        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([ao for _, ao in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)

        if not output_attenstion:
            return attention_output, None
        else:
            attention_prob = torch.stack([ap for ap, _ in attention_outputs])
            return attention_prob, attention_output


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, interm_dim: int, p: float):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, interm_dim)
        self.activation = nn.GELU()
        self.l2 = nn.Linear(interm_dim, hidden_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.dropout(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int,
        interm_dim: int,
        p: float,
        biase: bool,
    ):
        super().__init__()
        self.attention = MultiHead(hidden_dim, num_attention_heads, p, biase)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, interm_dim, p)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, output_attention=False):

        attention, attention_prob = self.attention(
            self.layer_norm_1(x), output_attention
        )
        x = x + attention
        mlp_output = self.mlp(self.layer_norm_2(x))
        x = x + mlp_output

        if output_attention:
            return x, None
        else:
            return x, attention_prob


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        hidden_dim: int,
        num_attention_heads: int,
        interm_dim: int,
        p: float,
        biase: bool,
    ):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            block = EncoderBlock(hidden_dim, num_attention_heads, interm_dim, p, biase)
            self.encoder_blocks.append(block)

    def forward(self, x, output_attention=False):

        all_attentions = []
        for block in self.encoder_blocks:
            x, attention_prob = block(x, output_attention)
            if output_attention:
                all_attentions.append(attention_prob)

        if not output_attention:
            return x, None
        else:
            return x, all_attentions


class ViT(nn.Module):

    def __init__(
        self,
        num_classes: int,
        img_size: int,
        in_chanels: int,
        hidden_dim: int,
        patch_size: int,
        num_blocks: int,
        num_attention_heads: int,
        interm_dim: int,
        p: float,
        biase: bool,
    ):
        super().__init__()
        self.embedings = Embedding(img_size, in_chanels, hidden_dim, patch_size, p)
        self.encoader = Encoder(
            num_blocks, hidden_dim, num_attention_heads, interm_dim, p, biase
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # self.apply(self._init_weights)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, output_attention=False):

        embedings = self.embedings(x)
        encoder_output, atttentions = self.encoader(embedings, output_attention)
        logits = self.classifier(encoder_output[:, 0])

        if not output_attention:
            return logits, None
        else:
            return logits, atttentions
