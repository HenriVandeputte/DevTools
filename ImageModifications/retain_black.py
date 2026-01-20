"""
Script to retain black lines in an image and make all other pixels transparent.
Uses OpenCV for image processing and optional AI-based edge detection.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def retain_black_lines_simple(image_path, output_path, threshold=30):
    """
    Simple method: Retain black pixels and make the rest transparent.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image with transparent background
        threshold: Threshold value for black detection (0-255)
                   Lower values = more restrictive to pure black
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB and get dimensions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create alpha channel (4th channel for RGBA)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Convert to grayscale to detect black lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create mask: identify black pixels (values close to 0)
    # Black lines typically have low values in all channels
    mask = cv2.inRange(gray, 0, threshold)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Set alpha channel: 255 (opaque) for black lines, 0 (transparent) for rest
    img_rgba[:, :, 3] = mask
    
    # Save the result
    cv2.imwrite(output_path, img_rgba)
    print(f"✓ Image saved to {output_path}")
    
    return img_rgba


def retain_black_lines_advanced(image_path, output_path, threshold=30, 
                                use_canny=True, canny_threshold1=50, canny_threshold2=150):
    """
    Advanced method: Uses Canny edge detection combined with black pixel detection.
    This helps detect both solid black lines and line edges.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image with transparent background
        threshold: Threshold for direct black pixel detection
        use_canny: Whether to use Canny edge detection
        canny_threshold1: Lower threshold for Canny
        canny_threshold2: Upper threshold for Canny
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Direct black pixel detection
    black_mask = cv2.inRange(gray, 0, threshold)
    
    # Method 2: Canny edge detection (optional)
    if use_canny:
        canny_edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
        # Combine black pixels with detected edges
        combined_mask = cv2.bitwise_or(black_mask, canny_edges)
    else:
        combined_mask = black_mask
    
    # Dilate to thicken lines slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # Set alpha channel
    img_rgba[:, :, 3] = combined_mask
    
    # Save the result
    cv2.imwrite(output_path, img_rgba)
    print(f"✓ Advanced image saved to {output_path}")
    
    return img_rgba


def retain_black_lines_invert(image_path, output_path, threshold=30):
    """
    Invert method: For images where you want to keep black lines but the 
    background might have some color/noise.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image with transparent background
        threshold: Threshold for black detection
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Check if each pixel has low values in all RGB channels (dark/black)
    b, g, r = cv2.split(img)
    
    # Pixel is black if all channels are below threshold
    black_mask = cv2.bitwise_and(cv2.bitwise_and(b < threshold, g < threshold), r < threshold)
    black_mask = black_mask.astype(np.uint8) * 255
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    img_rgba[:, :, 3] = black_mask
    
    cv2.imwrite(output_path, img_rgba)
    print(f"✓ Inverted method image saved to {output_path}")
    
    return img_rgba


def main():
    parser = argparse.ArgumentParser(
        description="Retain black lines in image and make rest transparent"
    )
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("-o", "--output", help="Path to save output image (default: output.png)")
    parser.add_argument("-m", "--method", choices=["simple", "advanced", "invert"], 
                       default="advanced", help="Processing method (default: advanced)")
    parser.add_argument("-t", "--threshold", type=int, default=30,
                       help="Black pixel threshold (0-255, default: 30)")
    parser.add_argument("--no-canny", action="store_true", 
                       help="Disable Canny edge detection for advanced method")
    
    args = parser.parse_args()
    
    # Set output path
    output_path = args.output
    if not output_path:
        input_stem = Path(args.input_image).stem
        output_path = f"{input_stem}_transparent.png"
    
    # Process based on selected method
    if args.method == "simple":
        retain_black_lines_simple(args.input_image, output_path, args.threshold)
    elif args.method == "advanced":
        retain_black_lines_advanced(args.input_image, output_path, args.threshold, 
                                   use_canny=not args.no_canny)
    elif args.method == "invert":
        retain_black_lines_invert(args.input_image, output_path, args.threshold)
    
    print(f"✓ Processing complete!")


if __name__ == "__main__":
    main()
