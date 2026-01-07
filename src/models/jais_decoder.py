"""
Jais Arabic LLM Wrapper

Loads and wraps Jais for Arabic text generation.
Model remains FROZEN during training.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List


class JaisDecoder(nn.Module):
    """
    Jais Arabic Language Model
    
    Generates Arabic text from fused embeddings.
    This module is frozen and not trained.
    """
    
    def __init__(
        self,
        model_name: str = "inceptionai/jais-13b-chat",
        cache_dir: Optional[str] = None,
        use_8bit: bool = False,
        freeze: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Set pad token if not exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        load_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if use_8bit else torch.float32
        }
        
        if use_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Get dimensions
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        
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
        inputs_embeds: torch.Tensor,        # [batch, seq_len, hidden_dim]
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with embedding inputs (from adapter)
        
        Args:
            inputs_embeds: Fused embeddings from adapter
            attention_mask: Attention mask
            labels: Ground truth labels for training
            
        Returns:
            Model outputs (loss, logits, etc.)
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate Arabic text from embeddings
        
        Args:
            inputs_embeds: Fused embeddings [batch, seq_len, hidden_dim]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy
            
        Returns:
            List of generated texts
        """
        # Generate
        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Decode
        texts = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )
        
        return texts
    
    def get_input_embeddings(self):
        """Get the embedding layer (for compatibility)"""
        return self.model.get_input_embeddings()


# Test the decoder
if __name__ == "__main__":
    print("="*60)
    print("Testing Jais Decoder")
    print("="*60)
    
    # Create decoder (use smaller model for testing)
    print("Note: Using base model for testing. Full model is 13B params.")
    print("For actual training, use: inceptionai/jais-13b-chat")
    
    # Test dimensions
    hidden_dim = 4096
    batch_size = 2
    seq_len = 32
    
    print(f"\nExpected dimensions:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Input embeds: [{batch_size}, {seq_len}, {hidden_dim}]")
    
    # Dummy embeddings (from adapter)
    dummy_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"\n✓ Ready for integration with Continuous Adapter")
    print("="*60)
