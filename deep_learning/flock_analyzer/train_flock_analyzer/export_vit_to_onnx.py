import torch
import torch.nn as nn
import os
from torchvision import transforms

# --- Vision Transformer (ViT) Components (Copied from train_flock_analyzer_ViT.py) ---

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # (B, E, H', W')
        x = x.flatten(2) # (B, E, N_patches)
        x = x.transpose(1, 2) # (B, N_patches, E)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FormationClassifier(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, num_classes=10, 
                 embed_dim=384, depth=6, num_heads=12, mlp_ratio=4., 
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., 
                 norm_layer=nn.LayerNorm, class_names=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.class_names = class_names if class_names is not None else []

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, 
                                          in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize positional embedding and CLS token
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def decode_prediction(self, pred):
        predicted_index = torch.argmax(pred, 1).item()
        if predicted_index < len(self.class_names):
            return self.class_names[predicted_index]
        else:
            return f"Unknown class index: {predicted_index}"

# --- ONNX Export Logic ---

if __name__ == "__main__":
    MODEL_PATH = "/home/saop/skyautonet_birdro/deep_learning/flock_analyzer/models/analyzer_ViT.pth"
    ONNX_OUTPUT_PATH = "/home/saop/skyautonet_birdro/deep_learning/flock_analyzer/models/analyzer_ViT.onnx"

    # Define model parameters based on train_flock_analyzer_ViT.py
    # You might need to adjust num_classes if it's not 10.
    # For this example, let's assume 5 classes for flock formations.
    # You should replace this with the actual number of classes your model was trained with.
    # If you don't know, you might need to inspect the training script's dataset loading.
    # For a dummy export, a placeholder like 5 or 10 is fine.
    NUM_CLASSES = 4 # Corrected based on error message: model was trained with 4 classes

    # Instantiate the model
    model = FormationClassifier(img_size=64, patch_size=8, in_channels=1, 
                                num_classes=NUM_CLASSES, 
                                embed_dim=384, depth=6, num_heads=12)
    
    # Load the trained state_dict
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded model from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()

    # Create a dummy input tensor
    # The input image is 64x64 grayscale, normalized to [-1, 1]
    # Batch size is 1 for export
    dummy_input = torch.randn(1, 1, 64, 64) # (Batch_size, Channels, Height, Width)

    # Export the model to ONNX
    try:
        torch.onnx.export(model,
                          dummy_input,
                          ONNX_OUTPUT_PATH,
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=11,          # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names = ['input'],   # the name of the input layer
                          output_names = ['output'], # the name of the output layer
                          dynamic_axes=None)
        print(f"Model successfully exported to ONNX at {ONNX_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")

