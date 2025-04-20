import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PARSeqLoss(nn.Module):
    """
    Loss function for PARSeq.

    Simple cross-entropy loss with optional label smoothing.
    """
    def __init__(self, ignore_index=-100, label_smoothing=0.1):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape [B, T, C] where C is the vocabulary size
            targets: Tensor of shape [B, T] containing target indices

        Returns:
            Loss value
        """
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )

class PARSeq(nn.Module):  # Changed from SimplePARSeq to PARSeq
    """
    Extremely simplified PARSeq model with robust NaN protection.
    """
    def __init__(
        self,
        vocab_size,
        max_label_length=25,
        img_size=(32, 128),
        embed_dim=256,  # Reduced from 384
        encoder_num_layers=4,  # Reduced from 12
        encoder_num_heads=4,  # Reduced from 6
        decoder_num_layers=3,  # Reduced from 6
        decoder_num_heads=4,  # Reduced from 8
        dropout=0.1,
        sos_idx=1,
        eos_idx=2,
        pad_idx=0,
        **kwargs  # To accept any extra arguments
    ):
        super().__init__()

        # Save config
        self.vocab_size = vocab_size
        self.max_label_length = max_label_length
        self.sos_id = sos_idx
        self.eos_id = eos_idx
        self.pad_id = pad_idx

        # Simple CNN encoder (ResNet-style)
        self.encoder = nn.Sequential(
            # Layer 1: 32x128 -> 16x64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: 16x64 -> 8x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: 8x32 -> 4x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: 4x16 -> 2x8
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Calculate sequence length from CNN output
        h, w = img_size[0] // 16, img_size[1] // 16  # After 4 stride-2 layers
        self.seq_len = h * w

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_label_length, embed_dim))

        # Transformer Decoder (PyTorch standard version)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=embed_dim * 2,  # Smaller feedforward size
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_num_layers
        )

        # Output projection
        self.classifier = nn.Linear(embed_dim, vocab_size)

        # Initialize weights carefully
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with extra care for numerical stability."""
        # Initialize embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.pos_encoder, -0.1, 0.1)

        # Initialize output projection carefully
        nn.init.uniform_(self.classifier.weight, -0.1, 0.1)
        nn.init.zeros_(self.classifier.bias)

        # Bias the model against predicting pad tokens
        with torch.no_grad():
            self.classifier.bias[self.pad_id] = -2.0

    def forward(self, imgs, targets=None):
        """Forward pass with NaN protection."""
        # Encode images
        features = self.encoder(imgs)

        # Reshape to sequence
        b, c, h, w = features.shape
        memory = features.view(b, c, -1).transpose(1, 2)  # [B, S, C]

        # Print encoder stats
        # if not self.training:
        #     print(f"Encoder output shape: {memory.shape}")
        #     print(f"Encoder min/max/mean: {memory.min().item():.4f}/{memory.max().item():.4f}/{memory.mean().item():.4f}")

        # In training mode
        if targets is not None:
            # Process targets: remove last token for input, remove first token for output
            tgt_in = targets[:, :-1]
            tgt_out = targets[:, 1:]

            # Create casual mask
            tgt_len = tgt_in.size(1)
            mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1).to(imgs.device)

            # Embeddings
            tgt_emb = self.embedding(tgt_in)

            # Add positional encoding
            pos = self.pos_encoder[:, :tgt_len].expand(b, -1, -1)
            tgt_emb = tgt_emb + pos

            # Run decoder with gradient clipping
            out = self.decoder(tgt_emb, memory, tgt_mask=mask)

            # Project to vocabulary with NaN protection
            logits = self.classifier(out)

            return {
                'logits': logits,
                'targets': tgt_out
            }

        # Inference mode
        else:
            # Start with SOS token
            curr_ids = torch.full((b, 1), self.sos_id, dtype=torch.long, device=imgs.device)

            # Store outputs for debug
            outputs = []

            # Auto-regressive generation
            for i in range(self.max_label_length):
                # Create mask (triangular matrix)
                tgt_len = curr_ids.size(1)
                mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1).to(imgs.device)

                # Embedding with positional encoding
                tgt_emb = self.embedding(curr_ids)
                pos = self.pos_encoder[:, :tgt_len].expand(b, -1, -1)
                tgt_emb = tgt_emb + pos

                # Run decoder
                out = self.decoder(tgt_emb, memory, tgt_mask=mask)

                # Get logits for last position
                logits = self.classifier(out[:, -1:])

                # Prevent NaN issues
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

                # Debug - print the first few logits
                # if i == 0:
                #     print(f"First step logits shape: {logits.shape}")
                #     top5 = torch.topk(logits[0, 0], 5)
                #     print(f"Top 5 tokens: {top5.indices.cpu().numpy()}, values: {top5.values.cpu().numpy()}")

                # Ensure we never predict PAD token
                logits[:, :, self.pad_id] = -1e10

                # Get next token
                next_token = torch.argmax(logits, dim=2)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                outputs.append(curr_ids.clone())

                # Stop at EOS
                if (next_token == self.eos_id).all():
                    break

            return {
                'logits': curr_ids,
                'outputs': outputs
            }
