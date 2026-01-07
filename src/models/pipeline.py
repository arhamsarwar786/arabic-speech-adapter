"""
Full Pipeline: ArTST → Adapter → Jais

Combines all components for end-to-end Arabic speech understanding.
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .artst_encoder import ArTSTEncoder
from .adapter import ContinuousAdapter
from .jais_decoder import JaisDecoder


class ArabicSpeechPipeline(nn.Module):
    """
    Complete pipeline for Arabic speech-to-text with emotion preservation
    
    Architecture:
        Audio → ArTST (frozen) → Adapter (trainable) → Jais (frozen) → Text
    """
    
    def __init__(
        self,
        artst_model: str = "MBZUAI/artst_asr_v2",
        jais_model: str = "inceptionai/jais-13b-chat",
        adapter_config: dict = None,
        cache_dir: Optional[str] = None,
        use_8bit_jais: bool = False
    ):
        super().__init__()
        
        # Default adapter config
        if adapter_config is None:
            adapter_config = {
                "speech_dim": 768,
                "hidden_dim": 4096,
                "num_prefix_tokens": 32,
                "num_attention_heads": 8,
                "intermediate_dim": 8192,
                "dropout": 0.1
            }
        
        print("Initializing Arabic Speech Pipeline...")
        print("="*60)
        
        # 1. ArTST Encoder (Frozen)
        print("Loading ArTST encoder...")
        self.artst = ArTSTEncoder(
            model_name=artst_model,
            cache_dir=cache_dir,
            freeze=True
        )
        print(f"✓ ArTST loaded: {self.artst.output_dim}D output")
        
        # 2. Continuous Adapter (Trainable)
        print("\nInitializing Continuous Adapter...")
        self.adapter = ContinuousAdapter(**adapter_config)
        print(f"✓ Adapter created: {self.adapter.get_num_trainable_params():,} params")
        
        # 3. Jais Decoder (Frozen)
        print("\nLoading Jais LLM...")
        self.jais = JaisDecoder(
            model_name=jais_model,
            cache_dir=cache_dir,
            use_8bit=use_8bit_jais,
            freeze=True
        )
        print(f"✓ Jais loaded: {self.jais.hidden_dim}D hidden")
        
        print("="*60)
        print("Pipeline ready!")
        print(f"Total trainable params: {self.get_num_trainable_params():,}")
        print("="*60)
    
    def forward(
        self,
        audio: torch.Tensor,                    # [batch, audio_samples]
        audio_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None   # For training
    ):
        """
        Full forward pass
        
        Args:
            audio: Raw audio waveform
            audio_attention_mask: Mask for padded audio
            labels: Ground truth text tokens (for training)
            
        Returns:
            Model outputs (loss, logits)
        """
        # 1. Encode speech
        speech_embeds = self.artst(audio, audio_attention_mask)
        
        # 2. Transform through adapter
        fused_embeds = self.adapter(speech_embeds)
        
        # 3. Generate with Jais
        outputs = self.jais(
            inputs_embeds=fused_embeds,
            labels=labels
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """
        Generate Arabic text from audio
        
        Args:
            audio: Raw audio [batch, samples]
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            
        Returns:
            Generated Arabic texts
        """
        # Encode speech
        speech_embeds = self.artst(audio)
        
        # Transform through adapter
        fused_embeds = self.adapter(speech_embeds)
        
        # Generate
        texts = self.jais.generate(
            inputs_embeds=fused_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return texts
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters (only adapter)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_adapter(self, path: str):
        """Save only the adapter weights"""
        torch.save({
            "adapter_state_dict": self.adapter.state_dict(),
            "config": {
                "speech_dim": self.adapter.speech_dim,
                "hidden_dim": self.adapter.hidden_dim,
                "num_prefix_tokens": self.adapter.num_prefix_tokens
            }
        }, path)
        print(f"✓ Adapter saved to {path}")
    
    def load_adapter(self, path: str):
        """Load adapter weights"""
        checkpoint = torch.load(path, map_location="cpu")
        self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        print(f"✓ Adapter loaded from {path}")


# Test the full pipeline
if __name__ == "__main__":
    import torchaudio
    
    print("\n" + "="*60)
    print("Testing Full Arabic Speech Pipeline")
    print("="*60 + "\n")
    
    # Create pipeline (will take time to download models)
    pipeline = ArabicSpeechPipeline()
    
    # Test with dummy audio
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds
    dummy_audio = torch.randn(batch_size, audio_length)
    
    print("\nTest Input:")
    print(f"  Audio shape: {dummy_audio.shape}")
    
    # Forward pass (training mode)
    outputs = pipeline(dummy_audio)
    print(f"\nTraining mode output:")
    print(f"  Logits shape: {outputs.logits.shape}")
    
    # Generation mode
    texts = pipeline.generate(dummy_audio, max_new_tokens=50)
    print(f"\nGeneration mode:")
    print(f"  Generated {len(texts)} texts")
    
    print("\n" + "="*60)
    print("✓ Pipeline test complete!")
    print("="*60)
