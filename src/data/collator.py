"""
Data Collator for batching Arabic speech samples
"""

import torch
from typing import List, Dict
from transformers import AutoTokenizer


class SpeechTextCollator:
    """
    Collates speech and text samples into batches
    
    Handles:
        - Audio padding
        - Text tokenization and padding
        - Attention masks
    """
    
    def __init__(
        self,
        tokenizer_name: str = "inceptionai/jais-13b-chat",
        max_audio_length: int = 480000,  # 30 seconds at 16kHz
        max_text_length: int = 512
    ):
        """
        Args:
            tokenizer_name: Jais tokenizer
            max_audio_length: Max audio samples
            max_text_length: Max text tokens
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            {
                'audio': [batch, max_audio_len],
                'audio_attention_mask': [batch, max_audio_len],
                'labels': [batch, max_text_len],
                'texts': List[str]
            }
        """
        # Extract audio and text
        audios = [item['audio'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Pad audio
        max_audio_len = min(
            max(audio.shape[0] for audio in audios),
            self.max_audio_length
        )
        
        padded_audios = []
        audio_masks = []
        
        for audio in audios:
            # Truncate if too long
            if audio.shape[0] > max_audio_len:
                audio = audio[:max_audio_len]
            
            # Pad if too short
            pad_len = max_audio_len - audio.shape[0]
            if pad_len > 0:
                audio = torch.cat([
                    audio,
                    torch.zeros(pad_len)
                ])
            
            # Create mask (1 for real audio, 0 for padding)
            mask = torch.ones(audio.shape[0])
            if pad_len > 0:
                mask[-pad_len:] = 0
            
            padded_audios.append(audio)
            audio_masks.append(mask)
        
        # Stack audio
        audio_batch = torch.stack(padded_audios)
        audio_mask_batch = torch.stack(audio_masks)
        
        # Tokenize text
        text_encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        
        return {
            'audio': audio_batch,
            'audio_attention_mask': audio_mask_batch,
            'labels': text_encoding.input_ids,
            'texts': texts
        }


# Test collator
if __name__ == "__main__":
    print("="*60)
    print("Testing Speech-Text Collator")
    print("="*60)
    
    # Create collator
    collator = SpeechTextCollator()
    print(f"✓ Collator created with tokenizer")
    
    # Create dummy batch
    batch = [
        {
            'audio': torch.randn(16000 * 2),  # 2 seconds
            'text': 'هذا نص تجريبي باللغة العربية',
            'audio_length': 16000 * 2,
            'text_length': 50,
            'dataset': 'test',
            'sample_id': '1'
        },
        {
            'audio': torch.randn(16000 * 3),  # 3 seconds
            'text': 'نص آخر أطول قليلاً',
            'audio_length': 16000 * 3,
            'text_length': 30,
            'dataset': 'test',
            'sample_id': '2'
        }
    ]
    
    # Collate
    collated = collator(batch)
    
    print(f"\nCollated batch:")
    print(f"  Audio: {collated['audio'].shape}")
    print(f"  Audio mask: {collated['audio_attention_mask'].shape}")
    print(f"  Labels: {collated['labels'].shape}")
    print(f"  Texts: {len(collated['texts'])} samples")
    
    print("="*60)
