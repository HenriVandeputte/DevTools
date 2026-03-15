#!/usr/bin/env python3
"""
Black-to-White Pixel Converter
Replaces black (and near-black) pixels with white while preserving
transparency and all other colors.
"""

import argparse
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: Pillow and numpy are required.")
    print("Install with: pip install Pillow numpy")
    sys.exit(1)


def select_image_files():
    """Open file dialog to select one or more image files."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_paths = filedialog.askopenfilenames(
        title="Select Image File(s) to Convert",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return list(file_paths)


def convert_blacks_to_white(image_path, threshold=10):
    """
    Convert black/near-black pixels to white in an image.

    Args:
        image_path: Path to the input image file
        threshold: Max R/G/B channel value (0-255) to treat as black. Default 10.

    Returns:
        (output_path, pixel_count) tuple
    """
    image_path = Path(image_path)
    img = Image.open(image_path)

    has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)

    # Work in RGBA so we always have an alpha channel to check
    rgba = img.convert('RGBA')
    data = np.array(rgba, dtype=np.uint16)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Mask: pixel is "black" if all channels <= threshold AND not fully transparent
    black_mask = (r <= threshold) & (g <= threshold) & (b <= threshold) & (a > 0)
    pixel_count = int(black_mask.sum())

    # Set matching pixels to white (keep original alpha)
    data[black_mask, 0] = 255
    data[black_mask, 1] = 255
    data[black_mask, 2] = 255

    result = Image.fromarray(data.astype(np.uint8), 'RGBA')

    # Determine output format and path
    if has_alpha:
        # Always PNG to preserve transparency
        output_path = image_path.with_name(image_path.stem + "_whites.png")
        result.save(output_path, 'PNG')
    else:
        # Drop alpha, keep original format
        output_path = image_path.with_name(image_path.stem + "_whites" + image_path.suffix)
        result.convert('RGB').save(output_path)

    return output_path, pixel_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert black pixels to white while preserving transparency and other colors."
    )
    parser.add_argument(
        "images",
        nargs='*',
        help="Image file path(s) to process (opens dialog if not provided)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        metavar="N",
        help="Max channel value (0-255) to treat as black. Default: 10"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI dialogs"
    )

    args = parser.parse_args()
    use_gui = not args.no_gui

    print("\n" + "="*60)
    print("BLACK-TO-WHITE PIXEL CONVERTER")
    print("="*60 + "\n")
    print(f"Threshold: {args.threshold} (pixels with R,G,B <= {args.threshold} become white)\n")

    # Collect input files
    if args.images:
        image_paths = args.images
    elif use_gui:
        print("Opening file selection dialog...\n")
        image_paths = select_image_files()
        if not image_paths:
            print("No files selected. Exiting.")
            sys.exit(0)
    else:
        print("Error: No image paths provided and GUI is disabled.")
        print("Usage: python blacks_to_white.py [image1 image2 ...] [--threshold N]")
        sys.exit(1)

    print(f"{len(image_paths)} file(s) selected.\n")

    succeeded = []
    failed = []

    for i, image_path in enumerate(image_paths, start=1):
        print(f"{'='*60}")
        print(f"FILE {i} of {len(image_paths)}: {Path(image_path).name}")
        print(f"{'='*60}")
        try:
            output_path, pixel_count = convert_blacks_to_white(image_path, args.threshold)
            print(f"  Pixels converted: {pixel_count:,}")
            print(f"  Saved to:         {output_path}\n")
            succeeded.append((image_path, output_path, pixel_count))
        except Exception as e:
            print(f"  Error: {e}\n")
            failed.append((image_path, str(e)))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Completed: {len(succeeded)} / {len(image_paths)}")
    for src, out, count in succeeded:
        print(f"  + {Path(src).name}  ->  {Path(out).name}  ({count:,} pixels changed)")
    for src, err in failed:
        print(f"  x {Path(src).name}  ->  ERROR: {err}")
    print("="*60 + "\n")

    if use_gui and (succeeded or failed):
        lines = [f"Converted {len(succeeded)} of {len(image_paths)} file(s).\n"]
        for src, out, count in succeeded:
            lines.append(f"+ {Path(src).name}\n  -> {out}\n  {count:,} pixels changed")
        for src, err in failed:
            lines.append(f"x {Path(src).name}\n  ERROR: {err}")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        if failed:
            messagebox.showwarning("Done (with errors)", "\n\n".join(lines))
        else:
            messagebox.showinfo("Done!", "\n\n".join(lines))
        root.destroy()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
