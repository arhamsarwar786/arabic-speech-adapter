"""
ArTST Speech Encoder Wrapper

Loads and wraps the pre-trained ArTST model for Arabic speech encoding.
Model remains FROZEN during training.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2Processor
from typing import Optional


class ArTSTEncoder(nn.Module):
    """
    ArTST (Arabic Transformer for Speech Translation) Encoder
    
    Pre-trained on Arabic speech, converts audio to continuous embeddings.
    This module is frozen and not trained.
    """
    
    def __init__(
        self,
        model_name: str = "MBZUAI/artst_asr_v2",
        cache_dir: Optional[str] = None,
        freeze: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Get output dimension
        self.output_dim = self.model.config.hidden_size
        
        # Freeze if specified
        if freeze:
            self.freeze()
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(
        self,
        audio: torch.Tensor,           # [batch, audio_length]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode audio to continuous embeddings
        
        Args:
            audio: Raw audio waveform [batch, samples]
            attention_mask: Optional mask for padded audio
            
        Returns:
            Speech embeddings [batch, time_frames, hidden_dim]
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(
                audio,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Get last hidden state
        speech_embeds = outputs.last_hidden_state
        
        return speech_embeds
    
    @torch.no_grad()
    def preprocess_audio(self, audio: torch.Tensor, sample_rate: int = 16000):
        """
        Preprocess audio using ArTST processor
        
        Args:
            audio: Audio tensor [samples] or [batch, samples]
            sample_rate: Sample rate (must be 16kHz for ArTST)
            
        Returns:
            Processed audio ready for model
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # ArTST expects 16kHz
        assert sample_rate == 16000, "ArTST requires 16kHz audio"
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values


# Test the encoder
if __name__ == "__main__":
    import torchaudio
    
    print("="*60)
    print("Testing ArTST Encoder")
    print("="*60)
    
    # Create encoder
    encoder = ArTSTEncoder()
    print(f"✓ Model loaded: {encoder.model_name}")
    print(f"✓ Output dimension: {encoder.output_dim}")
    print(f"✓ Parameters frozen: {not any(p.requires_grad for p in encoder.parameters())}")
    
    # Test with dummy audio
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds
    dummy_audio = torch.randn(batch_size, audio_length)
    
    # Encode
    speech_embeds = encoder(dummy_audio)
    
    print(f"\nTest Forward Pass:")
    print(f"  Input audio:  {dummy_audio.shape}")
    print(f"  Output embeds: {speech_embeds.shape}")
    print("="*60)
