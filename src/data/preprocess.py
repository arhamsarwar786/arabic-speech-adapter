"""
Preprocess downloaded datasets into format expected by dataset.py

Converts raw datasets to JSON with structure:
[
    {
        "audio_path": "relative/path/to/audio.wav",
        "text": "Arabic transcription",
        "id": "unique_id"
    },
    ...
]
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import torchaudio
import argparse


def preprocess_commonvoice(data_root: Path, split: str):
    """Preprocess Common Voice 17 Arabic"""
    print(f"\n{'='*60}")
    print(f"Preprocessing Common Voice ({split})")
    print('='*60)
    
    # Load from HuggingFace
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "ar",
        split=split,
        cache_dir=str(data_root / "raw" / "commonvoice")
    )
    
    samples = []
    audio_dir = data_root / "processed" / "commonvoice" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        # Save audio
        audio_filename = f"{split}_{idx:06d}.wav"
        audio_path = audio_dir / audio_filename
        
        # Get audio array and sample rate
        audio_array = item['audio']['array']
        sample_rate = item['audio']['sampling_rate']
        
        # Convert to tensor and save
        import torch
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
        torchaudio.save(str(audio_path), audio_tensor, sample_rate)
        
        # Create metadata
        samples.append({
            "audio_path": f"processed/commonvoice/{split}/{audio_filename}",
            "text": item['sentence'],
            "id": f"cv_{split}_{idx}"
        })
    
    # Save metadata
    metadata_path = data_root / "processed" / f"commonvoice_{split}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(samples)} samples to {metadata_path}")


def preprocess_qasr(data_root: Path, split: str):
    """Preprocess QASR dataset"""
    print(f"\n{'='*60}")
    print(f"Preprocessing QASR ({split})")
    print('='*60)
    
    # Load from HuggingFace
    dataset = load_dataset(
        "tarteel-ai/qasr",
        split=split,
        cache_dir=str(data_root / "raw" / "qasr")
    )
    
    samples = []
    audio_dir = data_root / "processed" / "qasr" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        audio_filename = f"{split}_{idx:06d}.wav"
        audio_path = audio_dir / audio_filename
        
        # Get audio
        audio_array = item['audio']['array']
        sample_rate = item['audio']['sampling_rate']
        
        import torch
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
        torchaudio.save(str(audio_path), audio_tensor, sample_rate)
        
        samples.append({
            "audio_path": f"processed/qasr/{split}/{audio_filename}",
            "text": item['text'],
            "id": f"qasr_{split}_{idx}"
        })
    
    metadata_path = data_root / "processed" / f"qasr_{split}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(samples)} samples to {metadata_path}")


def preprocess_mtedx(data_root: Path, split: str):
    """Preprocess mTEDx Arabic"""
    print(f"\n{'='*60}")
    print(f"Preprocessing mTEDx ({split})")
    print('='*60)
    
    # mTEDx is downloaded manually from OpenSLR
    raw_dir = data_root / "raw" / "mtedx" / "ar"
    
    if not raw_dir.exists():
        print(f"⚠ mTEDx raw data not found at {raw_dir}")
        print("Run download_arabic_datasets.py first!")
        return
    
    # Find audio files and transcripts
    samples = []
    audio_dir = data_root / "processed" / "mtedx" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for .wav files and corresponding .txt transcripts
    audio_files = sorted(raw_dir.rglob("*.wav"))
    
    for idx, audio_file in enumerate(tqdm(audio_files, desc="Processing")):
        # Find corresponding transcript
        txt_file = audio_file.with_suffix('.txt')
        if not txt_file.exists():
            continue
        
        # Read transcript
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            continue
        
        # Copy/convert audio to processed dir
        audio_filename = f"{split}_{idx:06d}.wav"
        audio_path = audio_dir / audio_filename
        
        # Load and resample to 16kHz
        waveform, sr = torchaudio.load(str(audio_file))
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        torchaudio.save(str(audio_path), waveform, 16000)
        
        samples.append({
            "audio_path": f"processed/mtedx/{split}/{audio_filename}",
            "text": text,
            "id": f"mtedx_{split}_{idx}"
        })
    
    metadata_path = data_root / "processed" / f"mtedx_{split}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(samples)} samples to {metadata_path}")


def preprocess_covost2(data_root: Path, split: str):
    """Preprocess CoVoST2 Arabic"""
    print(f"\n{'='*60}")
    print(f"Preprocessing CoVoST2 ({split})")
    print('='*60)
    
    # Load from HuggingFace
    dataset = load_dataset(
        "facebook/covost2",
        "ar_en",  # Arabic to English
        split=split,
        cache_dir=str(data_root / "raw" / "covost2")
    )
    
    samples = []
    audio_dir = data_root / "processed" / "covost2" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        audio_filename = f"{split}_{idx:06d}.wav"
        audio_path = audio_dir / audio_filename
        
        # Get audio
        audio_array = item['audio']['array']
        sample_rate = item['audio']['sampling_rate']
        
        import torch
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
        torchaudio.save(str(audio_path), audio_tensor, sample_rate)
        
        # Use source (Arabic) sentence
        samples.append({
            "audio_path": f"processed/covost2/{split}/{audio_filename}",
            "text": item['sentence'],  # Arabic text
            "id": f"covost2_{split}_{idx}"
        })
    
    metadata_path = data_root / "processed" / f"covost2_{split}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(samples)} samples to {metadata_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='Root data directory'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['commonvoice', 'qasr', 'mtedx', 'covost2'],
        help='Datasets to preprocess'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'validation', 'test'],
        help='Splits to process'
    )
    
    args = parser.parse_args()
    data_root = Path(args.data_root)
    
    # Create processed directory
    (data_root / "processed").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("ARABIC SPEECH DATASET PREPROCESSING")
    print("="*60)
    
    processors = {
        'commonvoice': preprocess_commonvoice,
        'qasr': preprocess_qasr,
        'mtedx': preprocess_mtedx,
        'covost2': preprocess_covost2
    }
    
    for dataset_name in args.datasets:
        if dataset_name not in processors:
            print(f"⚠ Unknown dataset: {dataset_name}")
            continue
        
        processor = processors[dataset_name]
        
        for split in args.splits:
            try:
                processor(data_root, split)
            except Exception as e:
                print(f"✗ Error processing {dataset_name} {split}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
