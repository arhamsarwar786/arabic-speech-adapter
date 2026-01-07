#!/bin/bash

##############################################################################
# Batch Inference Script
# 
# Process multiple audio files in batch mode (faster than Gradio for bulk)
#
# Usage:
#   bash scripts/7_batch_inference.sh <checkpoint> <input_dir> <output_dir>
#
# Example:
#   bash scripts/7_batch_inference.sh experiments/best_model/adapter_checkpoint.pt \
#                                     test_audios/ \
#                                     results/
##############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check arguments
if [ "$#" -lt 3 ]; then
    echo -e "${RED}Usage: $0 <checkpoint> <input_dir> <output_dir>${NC}"
    exit 1
fi

CHECKPOINT="$1"
INPUT_DIR="$2"
OUTPUT_DIR="$3"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  📊 Batch Inference${NC}"
echo -e "${BLUE}================================================${NC}"

# Check inputs
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}❌ Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Count audio files
NUM_FILES=$(find "$INPUT_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | wc -l)
echo -e "${GREEN}Found ${NUM_FILES} audio files${NC}"

# Run batch inference
echo -e "${GREEN}🚀 Starting batch inference...${NC}"

python -c "
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

sys.path.insert(0, str(Path.cwd()))
from src.models.pipeline import SpeechToTextPipeline
from omegaconf import OmegaConf

# Load config and model
config = OmegaConf.load('configs/training_config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Loading model on {device}...')
pipeline = SpeechToTextPipeline(
    artst_model_name=config.model.artst_model_name,
    jais_model_name=config.model.jais_model_name,
    adapter_hidden_dim=config.model.adapter_hidden_dim,
    adapter_num_heads=config.model.adapter_num_heads,
    num_prefix_tokens=config.model.num_prefix_tokens,
    device=device
)
pipeline.load_adapter('$CHECKPOINT')

# Get audio files
audio_files = list(Path('$INPUT_DIR').rglob('*.wav'))
audio_files += list(Path('$INPUT_DIR').rglob('*.mp3'))
audio_files += list(Path('$INPUT_DIR').rglob('*.flac'))

print(f'Processing {len(audio_files)} files...')

# Process each file
results = []
for audio_path in tqdm(audio_files):
    try:
        # Load audio
        audio, sr = sf.read(str(audio_path))
        
        # Resample if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            output = pipeline.generate(
                audio_tensor,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Save result
        output_path = Path('$OUTPUT_DIR') / f'{audio_path.stem}.txt'
        output_path.write_text(output, encoding='utf-8')
        
        results.append({
            'file': audio_path.name,
            'output': output,
            'status': 'success'
        })
        
    except Exception as e:
        print(f'Error processing {audio_path.name}: {e}')
        results.append({
            'file': audio_path.name,
            'output': '',
            'status': f'error: {e}'
        })

# Save summary
summary_path = Path('$OUTPUT_DIR') / 'summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Batch Inference Summary\n')
    f.write('=' * 50 + '\n\n')
    for r in results:
        f.write(f\"File: {r['file']}\n\")
        f.write(f\"Status: {r['status']}\n\")
        f.write(f\"Output: {r['output'][:100]}...\n\")
        f.write('-' * 50 + '\n\n')

print(f'✅ Done! Results saved to $OUTPUT_DIR')
print(f'Summary: {summary_path}')
"

echo ""
echo -e "${GREEN}✅ Batch inference complete!${NC}"
echo -e "Results saved to: ${OUTPUT_DIR}"
