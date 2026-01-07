"""
Arabic Speech Datasets for Training

Handles Common Voice, QASR, mTEDx, and CoVoST2 datasets.
"""

import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from typing import Dict, Optional
import json


class ArabicSpeechDataset(Dataset):
    """
    Unified dataset for all Arabic speech sources
    
    Supports:
        - Common Voice 17 (Arabic)
        - QASR (Quranic Arabic)
        - mTEDx (Arabic)
        - CoVoST2 (Arabic-English)
    """
    
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_audio_length: Optional[int] = None
    ):
        """
        Args:
            data_root: Root directory containing datasets
            dataset_name: One of ['commonvoice', 'qasr', 'mtedx', 'covost2']
            split: 'train', 'validation', or 'test'
            sample_rate: Target sample rate (16kHz for ArTST)
            max_audio_length: Maximum audio length in samples
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.split = split
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        # Load metadata
        self.samples = self._load_metadata()
        
        print(f"Loaded {dataset_name} ({split}): {len(self.samples)} samples")
    
    def _load_metadata(self):
        """Load dataset metadata"""
        metadata_path = self.data_root / "processed" / f"{self.dataset_name}_{self.split}.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}\n"
                f"Please run preprocessing first."
            )
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get one sample
        
        Returns:
            {
                'audio': Tensor [samples],
                'text': str (Arabic transcription),
                'audio_length': int,
                'text_length': int,
                'dataset': str,
                'sample_id': str
            }
        """
        sample = self.samples[idx]
        
        # Load audio
        audio_path = self.data_root / sample['audio_path']
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Squeeze to [samples]
        waveform = waveform.squeeze(0)
        
        # Truncate if too long
        if self.max_audio_length and waveform.shape[0] > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        return {
            'audio': waveform,
            'text': sample['text'],
            'audio_length': waveform.shape[0],
            'text_length': len(sample['text']),
            'dataset': self.dataset_name,
            'sample_id': sample.get('id', f"{idx}")
        }


class MultiDatasetLoader:
    """
    Combines multiple datasets with weighted sampling
    """
    
    def __init__(
        self,
        data_root: str,
        datasets: Dict[str, float],  # {name: weight}
        split: str = "train",
        sample_rate: int = 16000,
        max_audio_length: Optional[int] = None
    ):
        """
        Args:
            data_root: Root directory
            datasets: Dict of dataset names and weights
                     e.g., {'commonvoice': 0.4, 'qasr': 0.2, ...}
            split: Data split
            sample_rate: Target sample rate
            max_audio_length: Max audio length
        """
        self.datasets = {}
        self.weights = []
        self.total_samples = 0
        
        for name, weight in datasets.items():
            try:
                dataset = ArabicSpeechDataset(
                    data_root=data_root,
                    dataset_name=name,
                    split=split,
                    sample_rate=sample_rate,
                    max_audio_length=max_audio_length
                )
                self.datasets[name] = dataset
                self.weights.append(weight)
                self.total_samples += len(dataset)
            except FileNotFoundError as e:
                print(f"Warning: Skipping {name} - {e}")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"\nMulti-dataset loader created:")
        print(f"  Total datasets: {len(self.datasets)}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Weights: {dict(zip(self.datasets.keys(), self.weights))}")
    
    def get_combined_dataset(self):
        """Get combined dataset with weighted sampling"""
        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        
        # Concatenate all datasets
        datasets_list = list(self.datasets.values())
        combined = ConcatDataset(datasets_list)
        
        # Create sampling weights for each sample
        sample_weights = []
        for dataset, weight in zip(datasets_list, self.weights):
            sample_weights.extend([weight / len(dataset)] * len(dataset))
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(combined),
            replacement=True
        )
        
        return combined, sampler


# Test dataset loading
if __name__ == "__main__":
    print("="*60)
    print("Testing Arabic Speech Dataset")
    print("="*60)
    
    # This will fail until data is preprocessed
    try:
        dataset = ArabicSpeechDataset(
            data_root="/home/prof/Muneeb/project_files/data",
            dataset_name="commonvoice",
            split="train"
        )
        
        # Get one sample
        sample = dataset[0]
        print(f"\nSample structure:")
        print(f"  Audio shape: {sample['audio'].shape}")
        print(f"  Text: {sample['text'][:50]}...")
        print(f"  Dataset: {sample['dataset']}")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Dataset not ready: {e}")
        print("Run preprocessing first!")
    
    print("="*60)
