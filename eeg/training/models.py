
import torch
import torch.nn as nn

class EEGFlattenedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        seq_len: int = 1000,
        num_classes: int = 5,
        d_model: int = 128,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        
        self.flattened_dim = seq_len * d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        
        batch_size, seq_len, input_dim = x.shape
        
        x = self.input_projection(x)
        
        x = x + self.pos_embedding[:, :seq_len, :]
        
        x = x.reshape(batch_size, self.flattened_dim)
        
        logits = self.mlp(x)
        return logits

class EEGTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        seq_len: int = 1000,
        num_classes: int = 5,
        d_model: int = 128,
        num_heads: int = 8,
        feed_hidden_dim: int = 256
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)

        self.input_projection = nn.Linear(input_dim, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len + 1, d_model)
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feed_hidden_dim),
            nn.GELU(),
            nn.Linear(feed_hidden_dim, d_model)
        )
        
        self.classifier_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):

        batch_size, seq_len, input_dim = x.shape
        
        if seq_len != self.seq_len:
            raise ValueError(
                f"Expected seq_len {self.seq_len}, got {seq_len}"
            )
        
        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim {self.input_dim}, got {input_dim}"
            )

        x = self.input_projection(x)

        cls_token = self.cls_token.expand(batch_size, self.cls_token.shape[1], self.cls_token.shape[2])

        x = torch.cat([cls_token, x], dim = 1)
        
        x = x + self.pos_embedding[:, : seq_len + 1, :]

        normalized_x = self.attention_norm(x)
        
        attention_output, _ = self.self_attention(
            query = normalized_x,
            key = normalized_x,
            value = normalized_x,
            need_weights = False
        )
        
        x = x + attention_output
        
        normalized_x = self.feed_forward_norm(x)
        
        feed_forward_output = self.feed_forward(normalized_x)
        
        x = x + feed_forward_output

        cls_output = x[:, 0, :]

        logits = self.classifier_head(cls_output)

        return logits

class EEGConv2D(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 2,
        seq_len: int = 1000,
        num_classes: int = 5,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = conv1_channels,
                kernel_size = (5, input_dim),
                padding = (2, 0)
            ),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = conv1_channels,
                out_channels = conv2_channels,
                kernel_size = (5, 1),
                padding = (2, 0)
            ),
            nn.GELU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp = nn.Sequential(
            nn.Linear(conv2_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        
        batch_size, seq_len, input_dim = x.shape
        
        # [B, 1000, 2] -> [B, 1, 1000, 2]
        x = x.unsqueeze(1)
        
        # [B, 1, 1000, 2] -> [B, 32, 1000, 1]
        x = self.conv1(x)
        
        # [B, 32, 1000, 1] -> [B, 64, 1000, 1]
        x = self.conv2(x)
        
        # [B, 64, 1000, 1] -> [B, 64, 1, 1]
        x = self.pool(x)
        
        # [B, 64, 1, 1] -> [B, 64]
        x = x.reshape(batch_size, -1)
        
        # [B, 64] -> [B, 5]
        logits = self.mlp(x)
        return logits
    

class EEGSubtractiveConv2D(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 2,
        seq_len: int = 1000,
        num_classes: int = 5,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_dim: int = 256
    ):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(1.0))
        
        self.conv1_A = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = conv1_channels,
                kernel_size = (5, input_dim),
                padding = (2, 0)
            ),
            nn.GELU()
        )
        
        self.conv1_B = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = conv1_channels,
                kernel_size = (5, input_dim),
                padding = (2, 0)
            ),
            nn.GELU()
        )
        
        self.conv2_A = nn.Sequential(
            nn.Conv2d(
                in_channels = conv1_channels,
                out_channels = conv2_channels,
                kernel_size = (5, 1),
                padding = (2, 0)
            ),
            nn.GELU()
        )
        
        self.conv2_B = nn.Sequential(
            nn.Conv2d(
                in_channels = conv1_channels,
                out_channels = conv2_channels,
                kernel_size = (5, 1),
                padding = (2, 0)
            ),
            nn.GELU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp = nn.Sequential(
            nn.Linear(conv2_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        
        batch_size, seq_len, input_dim = x.shape
        
        # [B, 1000, 2] -> [B, 1, 1000, 2]
        x = x.unsqueeze(1)
        
        # [B, 1, 1000, 2] -> [B, 32, 1000, 1]
        A = self.conv1_A(x)
        B = self.conv1_B(x)
        x = A - self.lambda1 * B
        
        # [B, 32, 1000, 1] -> [B, 64, 1000, 1]
        A = self.conv2_A(x)
        B = self.conv2_B(x)
        x = A - self.lambda2 * B
        
        # [B, 64, 1000, 1] -> [B, 64, 1, 1]
        x = self.pool(x)
        
        # [B, 64, 1, 1] -> [B, 64]
        x = x.reshape(batch_size, -1)
        
        # [B, 64] -> [B, 5]
        logits = self.mlp(x)
        return logits
