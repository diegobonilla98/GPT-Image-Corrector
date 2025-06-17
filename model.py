import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PixelTransformerModel(nn.Module):
    """
    A Transformer model to process sequences of pixels.
    Input shape: (BATCH_SIZE, num_pixels, 3)
    Output shape: (BATCH_SIZE, num_pixels, 3)
    Input and output values are in the range [0, 1].
    """
    def __init__(self, num_pixels, input_dim=3, model_dim=512, num_layers=6, nhead=1, dim_feedforward=2048, dropout=0.1):
        """
        Args:
            num_pixels (int): The number of pixels in the input sequence (sequence length).
            input_dim (int): The dimension of each pixel (e.g., 3 for RGB).
            model_dim (int): The dimensionality of the model's embeddings and hidden layers.
                             Must be divisible by nhead.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            nhead (int): The number of heads in the multiheadattention models.
                         For this model, it's fixed to 1 as per requirement,
                         but kept as a parameter for standard TransformerEncoderLayer.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(PixelTransformerModel, self).__init__()
        self.model_dim = model_dim
        self.num_pixels = num_pixels

        # Input embedding: projects input_dim (e.g., 3 for RGB) to model_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=num_pixels)
        
        # Transformer Encoder Layer
        # Note: PyTorch's TransformerEncoderLayer expects nhead to divide model_dim.
        # If nhead is 1, this is always true.
        if model_dim % nhead != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by nhead ({nhead})")
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Important: input is (BATCH_SIZE, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: projects model_dim back to input_dim (e.g., 3 for RGB)
        self.output_projection = nn.Linear(model_dim, input_dim)
        
        # Sigmoid activation to ensure output is in [0, 1]
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers
        for lin_layer in [self.input_projection, self.output_projection]:
            nn.init.xavier_uniform_(lin_layer.weight)
            if lin_layer.bias is not None:
                nn.init.zeros_(lin_layer.bias)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [BATCH_SIZE, num_pixels, input_dim]
                 e.g., [BATCH_SIZE, 64, 3]
        
        Returns:
            Tensor, shape [BATCH_SIZE, num_pixels, input_dim]
        """
        # src shape: (BATCH_SIZE, num_pixels, input_dim)
        
        # Project input to model dimension
        src_projected = self.input_projection(src)  # Shape: (BATCH_SIZE, num_pixels, model_dim)
        src_projected = src_projected * math.sqrt(self.model_dim) # Scale embedding
        
        # So, we project, then permute for PE, then permute back for TransformerEncoder
        x = src_projected.permute(1, 0, 2) # (num_pixels, BATCH_SIZE, model_dim)
        x = self.pos_encoder(x)            # (num_pixels, BATCH_SIZE, model_dim)
        x = x.permute(1, 0, 2)             # (BATCH_SIZE, num_pixels, model_dim)
        
        # Pass through Transformer Encoder
        output = self.transformer_encoder(x) # Shape: (BATCH_SIZE, num_pixels, model_dim)
        
        # Project output back to input dimension
        output = self.output_projection(output) # Shape: (BATCH_SIZE, num_pixels, input_dim)
        
        # Apply sigmoid to get values in [0, 1]
        output = self.sigmoid(output) # Shape: (BATCH_SIZE, num_pixels, input_dim)
        
        return output
