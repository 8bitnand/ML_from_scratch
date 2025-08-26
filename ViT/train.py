from vit import ViT
import torch

num_classes = 10
img_size = 64
in_chanels = 3
hidden_dim = 512
patch_size = 8
num_blocks = 8
num_attention_heads = 8
interm_dim = 8 * 48
p = 0.0
biase = True
vit = ViT(
    num_classes,
    img_size,
    in_chanels,
    hidden_dim,
    patch_size,
    num_blocks,
    num_attention_heads,
    interm_dim,
    p,
    biase,
)

# print(vit)
x = torch.randn((1, 3, 64, 64))
print(vit(x))