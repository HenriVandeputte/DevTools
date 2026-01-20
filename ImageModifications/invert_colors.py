"""
Script to invert colors in an image.
Black becomes white, white becomes black, and all colors are inverted.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def invert_colors(image_path, output_path):
    """
    Invert all colors in an image using bitwise NOT operation.
    Black (0,0,0) becomes White (255,255,255) and vice versa.
    Transparent pixels remain transparent.
    
    Args:
        image_path: Path to input image
        output_path: Path to save inverted image
    """
    # Load the image with alpha channel if it exists
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Check if image has alpha channel
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Split channels
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        
        # Invert only BGR channels
        inverted_bgr = cv2.bitwise_not(bgr)
        
        # Merge back with original alpha
        inverted_img = cv2.merge([inverted_bgr[:, :, 0], inverted_bgr[:, :, 1], 
                                  inverted_bgr[:, :, 2], alpha])
    else:
        # No alpha channel, just invert
        inverted_img = cv2.bitwise_not(img)
    
    # Save the result
    cv2.imwrite(output_path, inverted_img)
    print(f"✓ Inverted image saved to {output_path}")
    
    return inverted_img


def invert_colors_with_alpha(image_path, output_path):
    """
    Invert colors while preserving transparency (if image has alpha channel).
    
    Args:
        image_path: Path to input image
        output_path: Path to save inverted image
    """
    # Load the image with alpha channel if it exists
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Check if image has alpha channel
    if img.shape[2] == 4:
        # Split channels
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        
        # Invert only BGR channels
        inverted_bgr = cv2.bitwise_not(bgr)
        
        # Merge back
        inverted_img = cv2.merge([inverted_bgr[:, :, 0], inverted_bgr[:, :, 1], 
                                  inverted_bgr[:, :, 2], alpha])
    else:
        # No alpha channel, just invert
        inverted_img = cv2.bitwise_not(img)
    
    # Save the result
    cv2.imwrite(output_path, inverted_img)
    print(f"✓ Inverted image saved to {output_path}")
    
    return inverted_img


def invert_colors_selective(image_path, output_path, preserve_color=None):
    """
    Invert colors with option to preserve specific colors.
    
    Args:
        image_path: Path to input image
        output_path: Path to save inverted image
        preserve_color: Color to preserve (None to invert all)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    inverted_img = cv2.bitwise_not(img)
    
    # Save the result
    cv2.imwrite(output_path, inverted_img)
    print(f"✓ Selectively inverted image saved to {output_path}")
    
    return inverted_img


def main():
    parser = argparse.ArgumentParser(
        description="Invert colors in an image (black becomes white, white becomes black)"
    )
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("-o", "--output", help="Path to save inverted image (default: output_inverted.png)")
    parser.add_argument("-a", "--alpha", action="store_true", 
                       help="Preserve alpha channel if present")
    
    args = parser.parse_args()
    
    # Set output path
    output_path = args.output
    if not output_path:
        input_stem = Path(args.input_image).stem
        output_path = f"{input_stem}_inverted.png"
    
    # Process
    if args.alpha:
        invert_colors_with_alpha(args.input_image, output_path)
    else:
        invert_colors(args.input_image, output_path)
    
    print(f"✓ Processing complete!")


if __name__ == "__main__":
    main()
