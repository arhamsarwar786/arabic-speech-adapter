"""
Gradio Interactive Demo for Arabic Speech-to-Text Adapter
Supports: Microphone input, file upload, real-time inference
"""

import os
import sys
import torch
import gradio as gr
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.pipeline import SpeechToTextPipeline
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GradioInferenceApp:
    """Gradio app for interactive inference"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/training_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Gradio app
        
        Args:
            checkpoint_path: Path to trained adapter checkpoint
            config_path: Path to training config
            device: Device to run inference on
        """
        self.device = device
        logger.info(f"Loading model on device: {device}")
        
        # Load config
        self.config = OmegaConf.load(config_path)
        
        # Initialize pipeline
        self.pipeline = SpeechToTextPipeline(
            artst_model_name=self.config.model.artst_model_name,
            jais_model_name=self.config.model.jais_model_name,
            adapter_hidden_dim=self.config.model.adapter_hidden_dim,
            adapter_num_heads=self.config.model.adapter_num_heads,
            num_prefix_tokens=self.config.model.num_prefix_tokens,
            use_8bit=self.config.training.get('use_8bit', False),
            device=device
        )
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.pipeline.load_adapter(checkpoint_path)
            logger.info("✅ Checkpoint loaded successfully!")
        else:
            logger.warning(f"⚠️  Checkpoint not found: {checkpoint_path}")
            logger.warning("Using untrained adapter (for testing only)")
        
        # Generation config
        self.generation_config = {
            'max_length': self.config.generation.max_length,
            'temperature': self.config.generation.temperature,
            'top_p': self.config.generation.top_p,
            'top_k': self.config.generation.top_k,
            'do_sample': self.config.generation.do_sample,
            'num_beams': self.config.generation.num_beams,
            'repetition_penalty': self.config.generation.repetition_penalty,
        }
        
        logger.info("🚀 App initialized successfully!")
    
    def process_audio(
        self,
        audio_input: Optional[Tuple[int, np.ndarray]],
        prompt: str = "",
        temperature: float = 0.7,
        max_length: int = 512,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[str, str]:
        """
        Process audio input and generate transcription
        
        Args:
            audio_input: Tuple of (sample_rate, audio_array) from Gradio
            prompt: Optional text prompt for the LLM
            temperature: Sampling temperature
            max_length: Maximum generation length
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Tuple of (transcription, status_message)
        """
        try:
            if audio_input is None:
                return "", "❌ Please provide audio input (microphone or file)"
            
            sample_rate, audio_array = audio_input
            logger.info(f"Processing audio: {audio_array.shape}, SR: {sample_rate}Hz")
            
            # Convert to float32 and normalize
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono if needed
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample to 16kHz if needed (ArTST expects 16kHz)
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
                import librosa
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0).to(self.device)
            
            # Update generation config
            gen_config = self.generation_config.copy()
            gen_config.update({
                'temperature': temperature,
                'max_length': max_length,
                'top_p': top_p,
                'top_k': top_k,
            })
            
            # Add prompt if provided
            if prompt.strip():
                gen_config['prompt_text'] = prompt.strip()
            
            # Generate transcription
            logger.info("Generating transcription...")
            with torch.no_grad():
                output = self.pipeline.generate(
                    audio_tensor,
                    **gen_config
                )
            
            transcription = output.strip()
            
            # Calculate audio duration
            duration = len(audio_array) / 16000.0
            
            status = f"✅ Success! Audio: {duration:.2f}s | Generated: {len(transcription)} chars"
            logger.info(status)
            
            return transcription, status
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return "", error_msg
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        with gr.Blocks(
            title="Arabic Speech-to-Text Adapter",
            theme=gr.themes.Soft(),
        ) as demo:
            
            gr.Markdown("""
            # 🎤 Arabic Speech-to-Text with Continuous Adapter
            
            Upload an audio file or record directly from your microphone to get Arabic transcriptions.
            
            **Model Architecture:**
            - 🎵 **Speech Encoder**: ArTST (MBZUAI/artst_asr_v2)
            - 🔄 **Adapter**: Continuous Adapter (Cross-Attention + Prefix Tuning + MLP)
            - 🤖 **Language Model**: Jais 13B (inceptionai/jais-13b-chat)
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Audio input
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="🎤 Audio Input (16kHz recommended)",
                        show_download_button=True,
                    )
                    
                    # Optional prompt
                    prompt_input = gr.Textbox(
                        label="📝 Optional Prompt (guide the LLM)",
                        placeholder="e.g., 'Transcribe the following Arabic speech:'",
                        lines=2,
                    )
                    
                    # Generation parameters
                    with gr.Accordion("⚙️ Generation Settings", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="🌡️ Temperature (higher = more creative)",
                        )
                        max_length_slider = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=512,
                            step=128,
                            label="📏 Max Length (tokens)",
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="🎯 Top-p (nucleus sampling)",
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="🔝 Top-k (sampling)",
                        )
                    
                    # Buttons
                    with gr.Row():
                        submit_btn = gr.Button("🚀 Transcribe", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ Clear", scale=1)
                
                with gr.Column(scale=2):
                    # Output
                    output_text = gr.Textbox(
                        label="📄 Transcription Output",
                        lines=10,
                        show_copy_button=True,
                    )
                    
                    # Status
                    status_text = gr.Textbox(
                        label="📊 Status",
                        lines=2,
                        interactive=False,
                    )
            
            # Examples
            gr.Markdown("### 📚 Example Usage")
            gr.Markdown("""
            1. **Microphone**: Click the microphone button and speak in Arabic
            2. **File Upload**: Upload an Arabic audio file (.wav, .mp3, .flac)
            3. **Prompt**: Optionally add a prompt like "Transcribe this speech:" (in Arabic or English)
            4. **Settings**: Adjust temperature (0.7 = balanced, 1.5 = creative)
            5. **Click Transcribe**: Get your transcription!
            """)
            
            # System info
            with gr.Accordion("ℹ️ System Information", open=False):
                gr.Markdown(f"""
                - **Device**: {self.device.upper()}
                - **Model**: {self.config.model.jais_model_name}
                - **Adapter Params**: ~{self.config.model.get('adapter_params', '10-50')}M
                - **PyTorch Version**: {torch.__version__}
                - **CUDA Available**: {torch.cuda.is_available()}
                """)
            
            # Event handlers
            submit_btn.click(
                fn=self.process_audio,
                inputs=[
                    audio_input,
                    prompt_input,
                    temperature_slider,
                    max_length_slider,
                    top_p_slider,
                    top_k_slider,
                ],
                outputs=[output_text, status_text],
            )
            
            clear_btn.click(
                fn=lambda: (None, "", "", ""),
                inputs=[],
                outputs=[audio_input, prompt_input, output_text, status_text],
            )
        
        return demo
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        auth: Optional[Tuple[str, str]] = None,
    ):
        """
        Launch the Gradio app
        
        Args:
            share: Create public link (useful for remote servers)
            server_name: Server host
            server_port: Server port
            auth: Optional (username, password) tuple for authentication
        """
        demo = self.create_interface()
        
        logger.info(f"🚀 Launching Gradio app on {server_name}:{server_port}")
        if share:
            logger.info("🌐 Creating public share link...")
        
        demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            auth=auth,
            show_error=True,
            show_api=True,
        )


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio demo for Arabic Speech Adapter")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained adapter checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--auth",
        type=str,
        nargs=2,
        metavar=("USERNAME", "PASSWORD"),
        help="Enable authentication (username password)"
    )
    
    args = parser.parse_args()
    
    # Create app
    app = GradioInferenceApp(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )
    
    # Launch
    auth_tuple = tuple(args.auth) if args.auth else None
    app.launch(
        share=args.share,
        server_port=args.port,
        auth=auth_tuple,
    )


if __name__ == "__main__":
    main()
