#!/usr/bin/env python3
"""
Video to Text Transcription using OpenAI Whisper
User-friendly GUI with file selection dialogs
Optimized for NVIDIA GPU
"""

import argparse
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import subprocess
import shutil

try:
    import whisper
except ImportError:
    print("Error: whisper is not installed.")
    print("Install it with: pip install openai-whisper")
    sys.exit(1)


def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return True, ffmpeg_path
        except Exception as e:
            return False, None
    return False, None


def check_gpu():
    """Check if CUDA GPU is available."""
    print("\n" + "="*70)
    print("GPU DETECTION")
    print("="*70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ CUDA Available: YES")
        print(f"✓ GPU Name: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU will be used for transcription (fp16=True)")
        return True, gpu_name
    else:
        print(f"✗ CUDA Available: NO")
        print(f"✗ PyTorch CUDA: {torch.version.cuda if torch.version.cuda else 'Not compiled with CUDA'}")
        print(f"✗ CPU will be used (much slower)")
        print(f"\nTo enable GPU:")
        print(f"  1. Ensure NVIDIA drivers are installed")
        print(f"  2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print(f"  3. Install PyTorch with CUDA: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False, None


def select_video_file():
    """Open file dialog to select video file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select Video File to Transcribe",
        filetypes=[
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def select_save_location(default_name):
    """Open file dialog to select save location."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.asksaveasfilename(
        title="Save Transcription As",
        defaultextension=".txt",
        initialfile=default_name,
        filetypes=[
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def transcribe_video(video_path, output_path=None, model_size="large", language=None, use_gui=True):
    """
    Transcribe a video file to text using Whisper.
    
    Args:
        video_path: Path to the video file
        output_path: Path for the output text file (optional)
        model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
        language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detect
        use_gui: Whether to use GUI dialogs
    """
    video_file = Path(video_path)
    
    # Validate input file
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check GPU availability
    gpu_available, gpu_name = check_gpu()
    if gpu_available:
        print(f"\n✓✓✓ GPU ACCELERATION ENABLED ✓✓✓")
        print(f"✓ Using: {gpu_name}")
        print("✓ Transcription will be 2-3x faster!")
        use_fp16 = True
    else:
        print(f"\n⚠⚠⚠ GPU NOT DETECTED - USING CPU ⚠⚠⚠")
        print("⚠ Transcription will be significantly slower")
        print("⚠ See GPU detection info above for troubleshooting")
        use_fp16 = False
    
    # Set output path using GUI if not provided
    if output_path is None and use_gui:
        default_name = video_file.stem + "_transcript.txt"
        output_path = select_save_location(default_name)
        if not output_path:  # User cancelled
            print("Save location selection cancelled.")
            return None
    elif output_path is None:
        output_path = video_file.with_suffix('.txt')
    
    output_path = Path(output_path)
    
    print("\n" + "="*70)
    print("TRANSCRIPTION SETTINGS")
    print("="*70)
    print(f"Video file:     {video_file}")
    print(f"Output file:    {output_path}")
    print(f"Model:          {model_size}")
    print(f"GPU Mode:       {'Enabled (fp16)' if use_fp16 else 'Disabled (CPU)'}")
    print(f"Language:       {'Auto-detect' if language is None else language}")
    print("="*70 + "\n")
    
    print("Loading Whisper model (this may take a moment)...")
    
    try:
        # Load the Whisper model
        model = whisper.load_model(model_size)
        
        print(f"✓ Model loaded successfully")
        print(f"\nStarting transcription...")
        print("This may take a while for a 3+ hour video...")
        print("Progress will be shown below:\n")
        
        # Transcribe the video with optimal settings for quality
        result = model.transcribe(
            str(video_file),
            language=language,
            verbose=True,
            fp16=use_fp16,  # GPU acceleration
            temperature=0.0,  # More deterministic output for best quality
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,  # CHANGED: Prevents hallucination loops
            initial_prompt=None,  # No initial prompt to avoid bias
            word_timestamps=False,  # Faster processing
            hallucination_silence_threshold=None  # Let model handle silences naturally
        )
        
        # Extract text from result
        transcription = result['text']
        
        # Save full transcription to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        # Save timestamped version
        '''
        timestamp_path = output_path.with_stem(output_path.stem + "_timestamped")
        with open(timestamp_path, 'w', encoding='utf-8') as f:
            for segment in result['segments']:
                start = segment['start']
                end = segment['end']
                text = segment['text']
                f.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")
        '''
        
        print("\n" + "="*70)
        print("✓ TRANSCRIPTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"✓ Full text saved to:       {output_path}")
        #print(f"✓ Timestamped version:      {timestamp_path}")
        print(f"✓ Total characters:         {len(transcription):,}")
        print(f"✓ Detected language:        {result.get('language', 'unknown')}")
        print(f"✓ Number of segments:       {len(result['segments'])}")
        print("="*70 + "\n")
        
        # Show success message box if using GUI
        if use_gui:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            messagebox.showinfo(
                "Transcription Complete!",
                f"Successfully transcribed video!\n\n"
                f"Saved to:\n{output_path}\n\n"
                f"Total characters: {len(transcription):,}"
            )
            root.destroy()
        
        return transcription
        
    except Exception as e:
        print(f"\n✗ Error during transcription: {e}")
        if use_gui:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            messagebox.showerror("Transcription Error", f"An error occurred:\n\n{str(e)}")
            root.destroy()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video to text using OpenAI Whisper with GUI (NVIDIA GPU optimized)"
    )
    parser.add_argument(
        "video_path",
        nargs='?',
        help="Path to the video file (optional - GUI will open if not provided)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output text file path (optional - GUI will open if not provided)",
        default=None
    )
    parser.add_argument(
        "-m", "--model",
        help="Whisper model size (larger = more accurate but slower)",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    )
    parser.add_argument(
        "-l", "--language",
        help="Language code (e.g., 'en', 'es', 'fr') or leave blank for auto-detect",
        default=None
    )
    parser.add_argument(
        "--no-gui",
        help="Disable GUI dialogs",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    use_gui = not args.no_gui
    
    print("\n" + "="*70)
    print("WHISPER VIDEO TRANSCRIPTION TOOL - BEST QUALITY MODE")
    print("="*70 + "\n")
    
    # Check FFmpeg first
    print("Checking system requirements...")
    ffmpeg_available, ffmpeg_path = check_ffmpeg()
    
    if not ffmpeg_available:
        error_msg = (
            "ERROR: FFmpeg is not installed or not found in PATH!\n\n"
            "FFmpeg is required to extract audio from video files.\n\n"
            "Please install FFmpeg:\n"
            "1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases\n"
            "   (Get 'ffmpeg-master-latest-win64-gpl.zip')\n"
            "2. Extract the ZIP file\n"
            "3. Add the 'bin' folder to your Windows PATH\n\n"
            "OR use the automatic installer:\n"
            "1. Install Chocolatey: https://chocolatey.org/install\n"
            "2. Run: choco install ffmpeg\n\n"
            "After installation, restart your terminal and try again."
        )
        print(error_msg)
        
        if use_gui:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            messagebox.showerror("FFmpeg Not Found", error_msg)
            root.destroy()
        
        sys.exit(1)
    else:
        print(f"✓ FFmpeg found: {ffmpeg_path}\n")
    
    # Check GPU
    gpu_available, gpu_name = check_gpu()
    if gpu_available:
        print(f"✓ NVIDIA GPU Detected: {gpu_name}")
        print("✓ GPU acceleration will be used for faster processing\n")
    else:
        print("⚠ No NVIDIA GPU detected - will use CPU (slower)\n")
    
    print("Model sizes:")
    print("  tiny      - Fastest, least accurate (~1GB)")
    print("  base      - Fast, decent accuracy (~1GB)")
    print("  small     - Balanced (~2GB)")
    print("  medium    - Slower, very accurate (~5GB)")
    print("  large     - Best quality (~10GB) [DEFAULT]")
    print("  large-v2  - Enhanced large model (~10GB)")
    print("  large-v3  - Latest, most accurate (~10GB)")
    print(f"\nUsing: {args.model.upper()} model for maximum quality\n")
    
    # Get video path
    video_path = args.video_path
    if not video_path and use_gui:
        print("Opening file selection dialog...\n")
        video_path = select_video_file()
        if not video_path:
            print("No file selected. Exiting.")
            sys.exit(0)
    elif not video_path:
        print("Error: No video path provided and GUI is disabled.")
        print("Use: python transcribe_video.py <video_file>")
        sys.exit(1)
    
    try:
        transcribe_video(video_path, args.output, args.model, args.language, use_gui)
    except Exception as e:
        print(f"Failed to transcribe video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()