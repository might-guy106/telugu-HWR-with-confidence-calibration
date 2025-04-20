import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class PatchEmbedding(nn.Module):
    """
    Converts input images to patch embeddings.

    Args:
        img_size: Tuple of (height, width)
        patch_size: Tuple of (height, width) for patch size
        in_channels: Number of input channels (1 for grayscale)
        embed_dim: Embedding dimension
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),  # (H, W)
        patch_size: Tuple[int, int] = (8, 4),   # (H, W) as per paper
        in_channels: int = 1,                   # Grayscale input
        embed_dim: int = 384                    # Embedding dimension
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        # Project image to patches
        # [B, C, H, W] -> [B, embed_dim, H', W'] where H' = H/patch_h, W' = W/patch_w
        x = self.proj(x)

        # Flatten spatial dimensions
        # [B, embed_dim, H', W'] -> [B, embed_dim, H'*W']
        x = x.flatten(2)

        # Transpose to get [B, num_patches, embed_dim]
        x = x.transpose(1, 2)

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to include bias in linear projections
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, E]
            key: [B, N_k, E]
            value: [B, N_v, E]
            attn_mask: [B, N_q, N_k] or None

        Returns:
            output: [B, N_q, E]
        """
        B, N_q, E = query.shape
        _, N_k, _ = key.shape

        # Linear projections and reshape for multi-head attention
        # [B, N, E] -> [B, N, h, d] -> [B, h, N, d]
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # [B, h, N_q, d] @ [B, h, d, N_k] -> [B, h, N_q, N_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attn_mask is not None:
            # Expand mask for batch and heads
            if attn_mask.dim() == 3:  # [B, N_q, N_k]
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N_q, N_k]
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum of values
        # [B, h, N_q, N_k] @ [B, h, N_k, d] -> [B, h, N_q, d]
        output = torch.matmul(attn_weights, v)

        # Reshape back to original dimensions
        # [B, h, N_q, d] -> [B, N_q, h, d] -> [B, N_q, E]
        output = output.transpose(1, 2).reshape(B, N_q, self.embed_dim)

        # Final linear projection
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):
    """
    Feed-forward network with residual connection.

    Args:
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        dropout: Dropout probability
    """
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 1536,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, E]

        Returns:
            output: [B, N, E]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer: self-attention + feed-forward.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for feed-forward network
        dropout: Dropout probability
        layer_norm_eps: Epsilon for layer normalization
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 8,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        # Self-attention layer
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, E]
            attn_mask: [B, N, N] or None

        Returns:
            output: [B, N, E]
        """
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward block
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x

        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in feed-forward network
        dropout: Dropout probability
        layer_norm_eps: Epsilon for layer normalization
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: Tuple[int, int] = (8, 4),
        in_channels: int = 1,
        embed_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 8,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Position embeddings
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])

        # Dropout for position embeddings
        self.pos_dropout = nn.Dropout(dropout)

        # Layer norm for output
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for transformer"""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            output: [B, N, E]
        """
        # Convert image to patch embeddings [B, N, E]
        x = self.patch_embed(x)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Process through transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # Apply final layer normalization
        x = self.norm(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with:
    1. Self-attention on target sequence
    2. Cross-attention between target and source sequence
    3. Feed-forward network

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for feed-forward network
        dropout: Dropout probability
        layer_norm_eps: Epsilon for layer normalization
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 8,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        # Self-attention layer
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-attention layer
        self.cross_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence [B, T, E]
            memory: Memory from encoder [B, N, E]
            tgt_mask: Target mask [B, T, T] or None
            memory_mask: Memory mask [B, T, N] or None

        Returns:
            output: [B, T, E]
        """
        # Self-attention block
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.dropout(tgt)
        tgt = residual + tgt

        # Cross-attention block
        residual = tgt
        tgt = self.norm2(tgt)
        tgt = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.dropout(tgt)
        tgt = residual + tgt

        # Feed-forward block
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = self.feed_forward(tgt)
        tgt = residual + tgt

        return tgt


class PermutationGenerator:
    """
    Generates token permutations for permutation language modeling.

    Args:
        max_seq_len: Maximum sequence length
        num_permutations: Number of permutations to generate
    """
    def __init__(self, max_seq_len: int = 35, num_permutations: int = 6):
        self.max_seq_len = max_seq_len
        self.num_permutations = num_permutations

    def get_permutations(self, batch_size: int, seq_len: int) -> List[torch.Tensor]:
        """
        Generate multiple permutations for each sequence in the batch.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            permutations: List of permutation tensors of shape [batch_size, seq_len]
        """
        permutations = []

        # Add sequential permutation
        sequential = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        permutations.append(sequential)

        # Add random permutations
        for _ in range(self.num_permutations - 1):
            batch_perms = []
            for b in range(batch_size):
                perm = torch.randperm(seq_len)
                batch_perms.append(perm)
            permutations.append(torch.stack(batch_perms))

        return permutations

    def get_causal_mask(self, length: int) -> torch.Tensor:
        """
        Generate causal mask for decoder self-attention.

        Args:
            length: Sequence length

        Returns:
            mask: Lower triangular matrix [length, length]
        """
        mask = torch.tril(torch.ones(length, length)).bool()
        return mask


class PARSeqDecoder(nn.Module):
    """
    PARSeq decoder with permutation language modeling.

    Args:
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in feed-forward network
        num_permutations: Number of permutations to use
        dropout: Dropout probability
        layer_norm_eps: Epsilon for layer normalization
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 35,
        embed_dim: int = 384,
        num_heads: int = 8,
        hidden_dim: int = 1536,
        num_permutations: int = 6,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_permutations = num_permutations

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim)  # +1 for [SOS] token

        # Position embedding
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))  # +1 for [SOS] token

        # Permutation generator
        self.permutation_generator = PermutationGenerator(max_seq_len, num_permutations)

        # Decoder layer
        self.decoder_layer = TransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size + 1)  # +1 for [SOS] token

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for transformer"""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward_permutation(
        self,
        tgt_input: torch.Tensor,
        memory: torch.Tensor,
        permutation: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for a single permutation.

        Args:
            tgt_input: Target input indices [B, T]
            memory: Memory from encoder [B, N, E]
            permutation: Permutation of target indices [B, T]
            tgt_padding_mask: Target padding mask [B, T] or None

        Returns:
            logits: Output logits [B, T, vocab_size]
        """
        B, T = tgt_input.shape

        # Permute target input based on the permutation
        # Gather along the sequence dimension for each batch item
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, T).to(tgt_input.device)
        tgt_permuted = tgt_input[batch_indices, permutation]

        # Add position embedding (not permuted)
        tgt_embedded = self.token_embedding(tgt_permuted)
        # Positions are sequential regardless of permutation
        positions = torch.arange(T).unsqueeze(0).expand(B, -1).to(tgt_permuted.device)
        tgt_embedded = tgt_embedded + self.position_embedding[:, :T]

        # Create causal attention mask for decoder self-attention
        causal_mask = self.permutation_generator.get_causal_mask(T).to(tgt_input.device)

        # Apply permutation to the causal mask
        # For each position i in the permuted sequence, we need to attend to all positions j
        # that come before i in the original ordering
        permuted_causal_mask = torch.zeros(B, T, T, dtype=torch.bool, device=tgt_input.device)
        for b in range(B):
            # Get the permutation for this batch item
            perm = permutation[b]
            # For each permuted position, create a mask that allows attending to positions that come before in the permutation
            for i in range(T):
                permuted_pos = perm[i]
                for j in range(T):
                    original_pos = perm[j]
                    # Can attend if original_pos comes before permuted_pos
                    if original_pos <= permuted_pos:
                        permuted_causal_mask[b, i, j] = True

        # Apply decoder layer
        tgt = tgt_embedded
        tgt = self.dropout(tgt)
        tgt = self.decoder_layer(tgt, memory, permuted_causal_mask, None)

        # Apply output projection
        tgt = self.norm(tgt)
        logits = self.output_projection(tgt)

        # Unpermute the logits to match the original sequence order
        # Use gather again but in the reverse direction
        inv_permutation = torch.zeros_like(permutation)
        for b in range(B):
            inv_permutation[b] = torch.argsort(permutation[b])

        logits_unpermuted = torch.zeros_like(logits)
        for b in range(B):
            logits_unpermuted[b] = logits[b, inv_permutation[b]]

        return logits_unpermuted

    def forward(
        self,
        tgt_input: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple permutations.

        Args:
            tgt_input: Target input indices [B, T]
            memory: Memory from encoder [B, N, E]
            tgt_padding_mask: Target padding mask [B, T] or None

        Returns:
            Dict with:
                logits: Dict of output logits for each permutation [B, T, vocab_size]
                permutations: List of permutation tensors [B, T]
        """
        B, T = tgt_input.shape

        # Generate permutations
        permutations = self.permutation_generator.get_permutations(B, T)

        # Forward pass for each permutation
        all_logits = {}
        for i, perm in enumerate(permutations):
            perm_name = "sequential" if i == 0 else f"random_{i}"
            all_logits[perm_name] = self.forward_permutation(tgt_input, memory, perm, tgt_padding_mask)

        return {
            "logits": all_logits,
            "permutations": permutations
        }


class PARSeq(nn.Module):
    """
    Permuted Autoregressive Sequence model for handwritten text recognition.

    Args:
        vocab_size: Size of vocabulary
        img_size: Input image size
        patch_size: Patch size for ViT
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        encoder_num_layers: Number of encoder layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in feed-forward networks
        max_seq_len: Maximum sequence length
        num_permutations: Number of permutations to use for training
        dropout: Dropout probability
        sos_idx: Start-of-sequence token index
    """
    def __init__(
        self,
        vocab_size: int,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: Tuple[int, int] = (8, 4),
        in_channels: int = 1,
        embed_dim: int = 384,
        encoder_num_layers: int = 12,
        num_heads: int = 8,
        hidden_dim: int = 1536,
        max_seq_len: int = 35,
        num_permutations: int = 6,
        dropout: float = 0.1,
        sos_idx: int = 1,  # Change default from 0 to 1
        pad_idx: int = 0   # Set padding token to blank token
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.num_permutations = num_permutations

        # Vision Transformer encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=encoder_num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # PARSeq decoder
        self.decoder = PARSeqDecoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_permutations=num_permutations,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        train: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]
            tgt: Optional target sequence [B, T]
            train: Whether in training mode

        Returns:
            Dict with model outputs
        """
        # Encode input image
        memory = self.encoder(x)
        batch_size = x.size(0)

        if train and tgt is not None:
            # Training mode with teacher forcing
            # Prepare inputs (add SOS token and remove last token)
            sos_tokens = torch.full((batch_size, 1), self.sos_idx, device=x.device)
            tgt_input = torch.cat([sos_tokens, tgt[:, :-1]], dim=1)

            # Decoder forward pass
            decoder_output = self.decoder(tgt_input, memory)
            return decoder_output
        else:
            # Inference mode with autoregressive decoding
            output_indices = []
            logit_scores = []
            batch_size = x.size(0)

            # Start with SOS token
            current_token = torch.full((batch_size, 1), self.sos_idx, device=x.device)

            # Autoregressive generation
            for i in range(self.max_seq_len):
                # Concatenate current token to output so far
                if i == 0:
                    tgt_input = current_token
                else:
                    tgt_input = torch.cat([tgt_input, current_token], dim=1)

                # Decoder forward pass (use only sequential permutation for inference)
                decoder_output = self.decoder(tgt_input, memory)
                logits = decoder_output["logits"]["sequential"]

                # Get next token (last position)
                next_token_logits = logits[:, -1, :]
                next_token_scores, next_token = torch.max(F.softmax(next_token_logits, dim=-1), dim=-1)

                # Add to output
                output_indices.append(next_token.unsqueeze(1))
                logit_scores.append(next_token_scores.unsqueeze(1))

                # Update current token for next iteration
                current_token = next_token.unsqueeze(1)

            # Combine outputs
            output_indices = torch.cat(output_indices, dim=1)
            logit_scores = torch.cat(logit_scores, dim=1)

            return {
                "predictions": output_indices,
                "scores": logit_scores
            }


class PermutationLoss(nn.Module):
    """
    Loss function for permutation language modeling.

    Args:
        ignore_index: Value to ignore in the target
        reduction: Reduction method for the loss
    """
    def __init__(self, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits_dict, targets, permutations):
        """
        Calculate permutation loss across all permutations.

        Args:
            logits_dict: Dictionary of logits for each permutation [B, T, V]
            targets: Target indices [B, T]
            permutations: List of permutation tensors [B, T]

        Returns:
            loss: Average loss across all permutations
        """
        total_loss = 0
        num_perms = len(permutations)

        for i, (name, logits) in enumerate(logits_dict.items()):
            # Get the permutation
            permutation = permutations[i]
            batch_size, seq_len = permutation.shape

            # Permute the targets
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len).to(targets.device)
            target_permuted = targets[batch_indices, permutation]

            # Calculate cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_permuted.reshape(-1),
                ignore_index=self.ignore_index,
                reduction=self.reduction
            )

            total_loss += loss

        # Return average loss across all permutations
        return total_loss / num_perms
