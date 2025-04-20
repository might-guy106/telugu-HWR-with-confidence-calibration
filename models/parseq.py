import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from itertools import permutations

class PatchEmbedding(nn.Module):
    """Convert input images to patch embeddings."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: Tuple[int, int] = (4, 8),
        in_channels: int = 1,
        embed_dim: int = 384
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Patch embeddings of shape [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        # Project patches
        x = self.proj(x)  # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        return x

class ViTEncoder(nn.Module):
    """Vision Transformer Encoder."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: Tuple[int, int] = (4, 8),
        in_channels: int = 1,
        embed_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks - explicitly calculate mlp dimension as an integer
        mlp_dim = int(mlp_ratio * embed_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads, mlp_dim, dropout
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get patch embeddings
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches + 1, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x  # Return all tokens including CLS

class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with pre-layer normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        # Ensure dimensions are proper integers
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        mlp_dim = int(mlp_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.dropout(attn_output)

        # MLP block
        norm_x = self.norm2(x)
        x = x + self.dropout(self.mlp(norm_x))

        return x

class MLP(nn.Module):
    """MLP with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Ensure dimensions are proper integers
        dim = int(dim)
        hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block supporting two-stream attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        # Self-attention for query stream
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_c = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward_stream(
        self,
        tgt: torch.Tensor,
        tgt_norm: torch.Tensor,
        tgt_kv: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass for a single stream."""
        # Self-attention
        tgt2, _ = self.self_attn(
            tgt_norm, tgt_kv, tgt_kv,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        # Cross-attention
        tgt2, _ = self.cross_attn(
            self.norm1(tgt), memory, memory
        )
        tgt = tgt + self.dropout2(tgt2)

        # MLP
        tgt2 = self.mlp(self.norm2(tgt))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(
        self,
        query: torch.Tensor,
        content: torch.Tensor,
        memory: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        content_mask: Optional[torch.Tensor] = None,
        content_key_padding_mask: Optional[torch.Tensor] = None,
        update_content: bool = True,
    ):
        """Forward pass with two-stream attention."""
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)

        # Update query with content as key-value
        query = self.forward_stream(
            query, query_norm, content_norm, memory,
            query_mask, content_key_padding_mask
        )

        # Update content if needed (not needed in the last layer)
        if update_content:
            content = self.forward_stream(
                content, content_norm, content_norm, memory,
                content_mask, content_key_padding_mask
            )

        return query, content

class TransformerDecoder(nn.Module):
    """Transformer decoder with multiple layers."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderBlock,
        num_layers: int,
        norm: nn.Module
    ):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query: torch.Tensor,
        content: torch.Tensor,
        memory: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        content_mask: Optional[torch.Tensor] = None,
        content_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass through all decoder layers."""
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = layer(
                query, content, memory,
                query_mask, content_mask, content_key_padding_mask,
                update_content=not last
            )

        query = self.norm(query)
        return query

class TokenEmbedding(nn.Module):
    """Embedding module for tokens."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply sqrt(d_model) scaling to embeddings."""
        return math.sqrt(self.embed_dim) * self.embedding(tokens)

class PARSeq(nn.Module):
    """
    PARSeq: Permuted Autoregressive Sequence model for handwritten text recognition.

    Implementation based on the paper:
    "Scene Text Recognition with Permuted Autoregressive Sequence Models"
    """

    def __init__(
        self,
        vocab_size: int,
        max_label_length: int = 25,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: Tuple[int, int] = (4, 8),
        in_channels: int = 1,
        embed_dim: int = 384,
        encoder_num_layers: int = 12,
        encoder_num_heads: int = 6,
        encoder_mlp_ratio: float = 4.0,
        decoder_num_layers: int = 6,
        decoder_num_heads: int = 6,
        decoder_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_permutations: int = 6,
        decode_ar: bool = True,
        refine_iters: int = 1,
        sos_idx: int = 1,  # Start of sequence token
        eos_idx: int = 2,  # End of sequence token
        pad_idx: int = 0,  # Padding token
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters
        self.num_permutations = num_permutations
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # Vision Transformer encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            dropout=dropout
        )

        # Decoder layer
        decoder_layer = TransformerDecoderBlock(
            embed_dim=embed_dim,
            num_heads=decoder_num_heads,
            mlp_dim=int(embed_dim * decoder_mlp_ratio),
            dropout=dropout
        )

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=decoder_num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # Token embedding
        self.text_embed = TokenEmbedding(vocab_size, embed_dim)

        # Output projection
        self.head = nn.Linear(embed_dim, vocab_size)

        # Position queries for decoding
        self.pos_queries = nn.Parameter(
            torch.zeros(1, max_label_length + 1, embed_dim)  # +1 for EOS
        )

        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights
        self._init_weights()

        # RNG for permutation generation
        self.rng = np.random.default_rng()

    def _init_weights(self):
        """Initialize weights with proper scaling to prevent NaN values"""
        # Use a smaller initialization scale
        nn.init.trunc_normal_(self.pos_queries, std=0.01)

        # Initialize embedding weights properly
        if hasattr(self.text_embed, 'embedding'):
            nn.init.normal_(self.text_embed.embedding.weight, mean=0.0, std=0.01)

        # Initialize output head carefully
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)  # Using smaller gain for stability

        # Initialize all other Linear layers with extra care
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _generate_permutations(self, batch_size: int, seq_length: int) -> List[torch.Tensor]:
        """Generate permutations for training."""
        # Max length of sequences in the batch
        max_len = seq_length

        # Start with canonical forward permutation
        perms = [torch.arange(max_len, device=self.pos_queries.device)]

        # Determine how many permutations to generate
        max_perms = math.factorial(max_len) // 2  # Only need half due to mirroring
        num_gen_perms = min(self.num_permutations // 2, max_perms)

        # Special handling for short sequences (generate all permutations)
        if max_len < 5:
            # Get all permutations
            perm_pool = torch.tensor(
                list(permutations(range(max_len), max_len)),
                device=self.pos_queries.device
            )

            # For some sequence lengths, we can divide the permutation space
            # in half (permutation and its reverse complement)
            if max_len == 4:  # Special case for length 4
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
                perm_pool = perm_pool[selector]
            else:
                # Just take first half of permutation pool
                perm_pool = perm_pool[:max_perms]

            # Skip the forward permutation which is already added
            perm_pool = perm_pool[1:]

            # Sample from pool if needed
            if num_gen_perms <= len(perm_pool):
                indices = self.rng.choice(len(perm_pool), size=num_gen_perms - 1, replace=False)
                sampled_perms = [perm_pool[i] for i in indices]
                perms.extend(sampled_perms)
            else:
                perms.extend(list(perm_pool))
        else:
            # For longer sequences, just generate random permutations
            for _ in range(num_gen_perms - 1):
                perm = torch.randperm(max_len, device=self.pos_queries.device)
                perms.append(perm)

        perms = torch.stack(perms)

        # Mirror permutations to get complement pairs
        mirrored = perms.flip(-1)
        perms = torch.cat([perms, mirrored])

        # Add batch dimension and expand
        perms = perms.unsqueeze(0).expand(batch_size, -1, -1)

        return perms

    def generate_causal_masks(self, permutation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate source and target attention masks for a permutation.

        Args:
            permutation: Tensor containing the permutation indices

        Returns:
            Tuple of (content_mask, query_mask) for the transformer decoder
        """
        # Get sequence size from the permutation
        batch_size, seq_len = permutation.shape[0], permutation.shape[1]

        # Create masks for each position in the permutation
        # Initialize with False (masked) everywhere
        content_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=permutation.device)
        query_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=permutation.device)

        # For each position, determine which positions it can attend to
        for i in range(seq_len):
            for j in range(seq_len):
                # Causal masking: position i can attend to position j if j <= i
                if j <= i:
                    content_mask[i, j] = True

                # Query masking: similar but exclude self-attention
                if j < i:  # Strictly less than for query mask
                    query_mask[i, j] = True

        return content_mask, query_mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images."""
        return self.encoder(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        tgt_query: Optional[torch.Tensor] = None,
        tgt_query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode with transformer decoder."""
        batch_size, seq_len = tgt.shape

        # BOS token embedding (null context)
        null_ctx = self.text_embed(tgt[:, :1])

        # Apply positional encoding to other tokens (not BOS)
        # We use fixed position queries, not learned positional encoding
        tgt_emb = self.pos_queries[:, :seq_len-1] + self.text_embed(tgt[:, 1:])

        # Concatenate BOS embedding and positional embeddings
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))

        # Position queries for the decoder
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :seq_len].expand(batch_size, -1, -1)
        tgt_query = self.dropout(tgt_query)

        # Make sure masks match sequence dimensions
        if tgt_mask is not None and tgt_mask.shape[0] != seq_len:
            # Resize mask if needed
            tgt_mask = tgt_mask[:seq_len, :seq_len]

        if tgt_query_mask is not None and tgt_query_mask.shape[0] != seq_len:
            # Resize query mask if needed
            tgt_query_mask = tgt_query_mask[:seq_len, :seq_len]

        # Run decoder with two-stream attention
        return self.decoder(
            tgt_query, tgt_emb, memory,
            tgt_query_mask, tgt_mask, tgt_padding_mask
        )

    def forward(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # Encode images - adding a small epsilon for numerical stability
        memory = self.encode(images)
        memory = memory + 1e-6  # Small constant to prevent exact zeros

        # Training mode with targets
        if targets is not None:
            # Get target sequence length
            seq_len = targets.size(1)

            # Setup inputs: tgt_input has BOS and characters, target_output has characters and EOS
            tgt_input = targets[:, :-1]  # Remove EOS token
            tgt_output = targets[:, 1:]  # Remove SOS token

            # Standard causal mask for autoregressive training
            mask_size = tgt_input.size(1)
            mask = torch.tril(torch.ones(mask_size, mask_size, device=targets.device)).bool()

            # Apply decoder with gradient clipping
            with torch.autograd.set_detect_anomaly(True):  # Debug NaNs
                # Apply decoder
                tgt_out = self.decode(tgt_input, memory, tgt_mask=mask)

                # Get logits and apply scaling for numerical stability
                logits = self.head(tgt_out)

                # Apply layer norm before returning logits - helps with stability
                logits = F.layer_norm(logits, normalized_shape=[logits.size(-1)])

            return {
                "logits": logits,
                "targets": tgt_output
            }

        # Inference mode (autoregressive decoding) - simplified
        else:
            batch_size = images.shape[0]
            max_length = 35  # Fixed max length for safety

            # Initialize with only SOS token (1)
            current_ids = torch.full((batch_size, 1), 1, dtype=torch.long, device=images.device)
            all_ids = [current_ids]

            # Simple auto-regressive loop for inference
            for _ in range(max_length):
                # Create mask
                cur_len = current_ids.size(1)
                mask = torch.tril(torch.ones(cur_len, cur_len, device=images.device)).bool()

                # Decode current sequence
                output = self.decode(current_ids, memory, tgt_mask=mask)

                # Predict next token (last position only)
                next_token_logits = self.head(output[:, -1:])
                next_token = next_token_logits.argmax(dim=-1)

                # Add to predictions
                all_ids.append(next_token)
                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Stop if all sequences have EOS token
                if (next_token == self.eos_idx).all():
                    break

            # Return raw token ids for decoding
            return {"logits": current_ids}

class PARSeqLoss(nn.Module):
    """
    Loss function for PARSeq.

    Calculates cross-entropy loss for each permutation and averages.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits_list: List[torch.Tensor],
        targets: torch.Tensor,
        permutations: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits_list: List of logits for each permutation [B, T, vocab_size]
            targets: Target tokens [B, T]
            permutations: List of permutation tensors [B, num_perms, T]

        Returns:
            Average loss across all permutations
        """
        batch_size = targets.size(0)
        total_loss = 0
        total_elements = 0

        # First 2 permutations: standard and reverse order
        for i in range(min(2, len(logits_list))):
            # For standard forward and reverse permutations, use original targets
            logits = logits_list[i]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.ignore_index,
                reduction='sum'
            )

            # Count valid elements (not ignored)
            elements = (targets != self.ignore_index).sum().item()
            total_loss += loss
            total_elements += elements

        # For other permutations, modify targets to ignore EOS tokens
        if len(logits_list) > 2:
            # Create a mask where EOS tokens will be ignored in additional permutations
            targets_no_eos = targets.clone()
            targets_no_eos[targets == 2] = self.ignore_index  # Change EOS to ignore_index

            for i in range(2, len(logits_list)):
                logits = logits_list[i]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets_no_eos.reshape(-1),
                    ignore_index=self.ignore_index,
                    reduction='sum'
                )

                # Count valid elements (not ignored)
                elements = (targets_no_eos != self.ignore_index).sum().item()
                total_loss += loss
                total_elements += elements

        # Return average loss
        if total_elements > 0:
            return total_loss / total_elements
        return total_loss  # Will be 0 if no valid elements
