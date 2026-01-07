"""
Continuous Adapter Module

Architecture:
    Speech Embeddings → Cross Attention → Prefix Tuning → MLP → Fused Embeddings

This adapter bridges the gap between ArTST speech encoder and Jais LLM.
Only this module is trainable; ArTST and Jais remain frozen.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention between speech embeddings and text prefix.
    Allows the model to selectively focus on relevant speech features.
    """
    
    def __init__(
        self,
        speech_dim: int = 768,      # ArTST output dimension
        hidden_dim: int = 4096,     # Jais hidden dimension
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Project speech to hidden dimension
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        speech_embeds: torch.Tensor,  # [batch, seq_len, speech_dim]
        prefix_embeds: torch.Tensor    # [batch, prefix_len, hidden_dim]
    ) -> torch.Tensor:
        """
        Args:
            speech_embeds: Speech features from ArTST
            prefix_embeds: Learnable prefix tokens
            
        Returns:
            Attended features [batch, prefix_len, hidden_dim]
        """
        # Project speech to hidden dimension
        speech_proj = self.speech_proj(speech_embeds)  # [batch, seq_len, hidden_dim]
        
        # Cross attention: prefix attends to speech
        attn_output, _ = self.attention(
            query=prefix_embeds,      # What we want to learn
            key=speech_proj,          # Speech features
            value=speech_proj         # Speech features
        )
        
        # Residual + norm
        output = self.norm(prefix_embeds + attn_output)
        
        return output


class PrefixTuning(nn.Module):
    """
    Learnable prefix tokens that act as "soft prompts" for the LLM.
    These tokens condition the LLM on the speech input.
    """
    
    def __init__(
        self,
        num_prefix_tokens: int = 32,
        hidden_dim: int = 4096
    ):
        super().__init__()
        
        self.num_prefix_tokens = num_prefix_tokens
        
        # Learnable prefix embeddings
        self.prefix_embeds = nn.Parameter(
            torch.randn(1, num_prefix_tokens, hidden_dim) * 0.02
        )
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Returns prefix tokens expanded for batch
        
        Returns:
            [batch_size, num_prefix_tokens, hidden_dim]
        """
        return self.prefix_embeds.expand(batch_size, -1, -1)


class AdapterMLP(nn.Module):
    """
    MLP to further transform the fused embeddings.
    Projects attended speech features into the exact space Jais expects.
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        intermediate_dim: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            
        Returns:
            Transformed embeddings [batch, seq_len, hidden_dim]
        """
        return self.norm(x + self.net(x))


class ContinuousAdapter(nn.Module):
    """
    Full Continuous Adapter Module
    
    Combines:
        1. Prefix Tuning (learnable soft prompts)
        2. Cross Attention (attend speech → prefix)
        3. MLP (final projection)
    
    This creates "fused embeddings" that Jais can understand while
    preserving speech information (emotions, prosody, etc.)
    """
    
    def __init__(
        self,
        speech_dim: int = 768,          # ArTST output
        hidden_dim: int = 4096,         # Jais hidden
        num_prefix_tokens: int = 32,
        num_attention_heads: int = 8,
        intermediate_dim: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.speech_dim = speech_dim
        self.hidden_dim = hidden_dim
        self.num_prefix_tokens = num_prefix_tokens
        
        # Components
        self.prefix_tuning = PrefixTuning(num_prefix_tokens, hidden_dim)
        
        self.cross_attention = CrossAttentionLayer(
            speech_dim=speech_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.mlp = AdapterMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout
        )
        
    def forward(
        self,
        speech_embeds: torch.Tensor,           # [batch, seq_len, speech_dim]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transform speech embeddings into fused embeddings for Jais
        
        Args:
            speech_embeds: ArTST output embeddings
            attention_mask: Optional mask for padded sequences
            
        Returns:
            Fused embeddings [batch, num_prefix_tokens, hidden_dim]
        """
        batch_size = speech_embeds.size(0)
        
        # 1. Get learnable prefix tokens
        prefix_embeds = self.prefix_tuning(batch_size)
        
        # 2. Cross attention: prefix attends to speech
        attended_embeds = self.cross_attention(speech_embeds, prefix_embeds)
        
        # 3. MLP transformation
        fused_embeds = self.mlp(attended_embeds)
        
        return fused_embeds
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test the adapter
    batch_size = 4
    seq_len = 150  # ~3 seconds of audio at 50 fps
    speech_dim = 768
    
    # Create adapter
    adapter = ContinuousAdapter(
        speech_dim=speech_dim,
        hidden_dim=4096,
        num_prefix_tokens=32
    )
    
    # Dummy speech embeddings (from ArTST)
    speech_embeds = torch.randn(batch_size, seq_len, speech_dim)
    
    # Forward pass
    fused_embeds = adapter(speech_embeds)
    
    print("="*60)
    print("Continuous Adapter Test")
    print("="*60)
    print(f"Input (Speech):  {speech_embeds.shape}")
    print(f"Output (Fused):  {fused_embeds.shape}")
    print(f"Trainable params: {adapter.get_num_trainable_params():,}")
    print("="*60)
    
    # Expected output:
    # Input:  [4, 150, 768]   - Speech embeddings
    # Output: [4, 32, 4096]   - Fused embeddings for Jais
    # Params: ~10-50M         - Much smaller than Q-Former
