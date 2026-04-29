#!/usr/bin/env python3
"""
MP4 to GIF Converter
User-friendly GUI with file selection dialogs.
Uses FFmpeg with a two-pass palette workflow for high-quality GIFs.
"""

import argparse
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox


def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return True, ffmpeg_path
        except Exception:
            return False, None
    return False, None


def select_video_files():
    """Open file dialog to select one or more video files."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_paths = filedialog.askopenfilenames(
        title="Select Video File(s) to Convert to GIF",
        filetypes=[
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    return list(file_paths)


def convert_to_gif(video_path, output_path=None, fps=15, width=480, start=None, duration=None):
    """
    Convert a video file to an animated GIF using a two-pass palette workflow.

    Args:
        video_path: Path to the source video.
        output_path: Path for the output .gif (auto-derived next to source if None).
        fps: Frames per second for the GIF.
        width: Output width in pixels (height auto-scaled, preserving aspect ratio).
               Pass -1 to keep the source width.
        start: Optional start time (seconds or HH:MM:SS) to begin conversion.
        duration: Optional duration (seconds or HH:MM:SS) to include.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = video_file.with_suffix(".gif")
    output_path = Path(output_path)

    vf_filter = f"fps={fps},scale={width}:-1:flags=lanczos"

    print("\n" + "=" * 70)
    print("CONVERSION SETTINGS")
    print("=" * 70)
    print(f"Source:    {video_file}")
    print(f"Output:    {output_path}")
    print(f"FPS:       {fps}")
    print(f"Width:     {width}px (height auto)")
    if start is not None:
        print(f"Start:     {start}")
    if duration is not None:
        print(f"Duration:  {duration}")
    print("=" * 70 + "\n")

    trim_args = []
    if start is not None:
        trim_args += ["-ss", str(start)]
    if duration is not None:
        trim_args += ["-t", str(duration)]

    with tempfile.TemporaryDirectory() as tmpdir:
        palette = Path(tmpdir) / "palette.png"

        print("Pass 1/2: generating color palette...")
        palette_cmd = [
            "ffmpeg", "-y",
            *trim_args,
            "-i", str(video_file),
            "-vf", f"{vf_filter},palettegen=stats_mode=diff",
            str(palette),
        ]
        subprocess.run(palette_cmd, check=True)

        print("\nPass 2/2: encoding GIF with palette...")
        gif_cmd = [
            "ffmpeg", "-y",
            *trim_args,
            "-i", str(video_file),
            "-i", str(palette),
            "-lavfi", f"{vf_filter} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            str(output_path),
        ]
        subprocess.run(gif_cmd, check=True)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Saved to:  {output_path}")
    print(f"Size:      {size_mb:.2f} MB")
    print("=" * 70 + "\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 (or other) videos to animated GIFs.")
    parser.add_argument("video_path", nargs="?", help="Path to video (optional, GUI opens if omitted)")
    parser.add_argument("-o", "--output", default=None, help="Output .gif path")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default 15)")
    parser.add_argument("-w", "--width", type=int, default=480, help="Output width in px, -1 to keep source (default 480)")
    parser.add_argument("--start", default=None, help="Start time, e.g. 5 or 00:00:05")
    parser.add_argument("--duration", default=None, help="Duration, e.g. 10 or 00:00:10")
    parser.add_argument("--no-gui", action="store_true", help="Disable file dialogs")
    args = parser.parse_args()

    use_gui = not args.no_gui

    print("\n" + "=" * 70)
    print("MP4 TO GIF CONVERTER")
    print("=" * 70 + "\n")

    print("Checking system requirements...")
    ffmpeg_available, ffmpeg_path = check_ffmpeg()
    if not ffmpeg_available:
        error_msg = (
            "ERROR: FFmpeg is not installed or not on PATH.\n\n"
            "Install it from https://github.com/BtbN/FFmpeg-Builds/releases\n"
            "or via Chocolatey: choco install ffmpeg\n"
        )
        print(error_msg)
        if use_gui:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            messagebox.showerror("FFmpeg Not Found", error_msg)
            root.destroy()
        sys.exit(1)
    print(f"FFmpeg found: {ffmpeg_path}\n")

    if args.video_path:
        video_paths = [args.video_path]
    elif use_gui:
        print("Opening file selection dialog...\n")
        video_paths = select_video_files()
        if not video_paths:
            print("No file(s) selected. Exiting.")
            sys.exit(0)
    else:
        print("Error: No video path provided and GUI is disabled.")
        sys.exit(1)

    if args.output and len(video_paths) > 1:
        print("Warning: --output is ignored for multi-file batches.\n")
        args.output = None

    succeeded, failed = [], []
    for i, path in enumerate(video_paths, start=1):
        print(f"\n{'=' * 70}")
        print(f"FILE {i} of {len(video_paths)}: {Path(path).name}")
        print(f"{'=' * 70}")
        try:
            out = convert_to_gif(
                path,
                output_path=args.output if len(video_paths) == 1 else None,
                fps=args.fps,
                width=args.width,
                start=args.start,
                duration=args.duration,
            )
            succeeded.append((path, out))
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed for '{path}' (exit code {e.returncode})")
            failed.append((path, f"FFmpeg exit {e.returncode}"))
        except Exception as e:
            print(f"Failed for '{path}': {e}")
            failed.append((path, str(e)))

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"  Completed: {len(succeeded)} / {len(video_paths)}")
    for src, out in succeeded:
        print(f"  OK  {Path(src).name}  ->  {Path(out).name}")
    for src, err in failed:
        print(f"  ERR {Path(src).name}  ->  {err}")
    print("=" * 70 + "\n")

    if use_gui and (succeeded or failed):
        lines = [f"Completed {len(succeeded)} of {len(video_paths)} file(s).\n"]
        for src, out in succeeded:
            lines.append(f"OK  {Path(src).name}\n   -> {out}")
        for src, err in failed:
            lines.append(f"ERR {Path(src).name}\n   {err}")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        if failed:
            messagebox.showwarning("Batch Complete (with errors)", "\n\n".join(lines))
        else:
            messagebox.showinfo("Batch Complete!", "\n\n".join(lines))
        root.destroy()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
