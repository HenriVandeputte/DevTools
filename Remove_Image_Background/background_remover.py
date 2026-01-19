import tkinter as tk
from tkinter import filedialog
from rembg import remove, new_session
from PIL import Image
import os
import onnxruntime as ort

def check_gpu_availability():
    """Check if CUDA GPU is available for onnxruntime."""
    providers = ort.get_available_providers()
    print("\nAvailable ONNX Runtime providers:")
    for provider in providers:
        print(f"  - {provider}")

    if 'CUDAExecutionProvider' in providers:
        print("\nCUDA GPU is available and will be used for acceleration!")
        return True
    else:
        print("\nWarning: CUDA GPU not available, falling back to CPU")
        return False

def select_image():
    """Open file dialog to select an image file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.svg"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("SVG files", "*.svg"),
            ("All files", "*.*")
        ]
    )

    return file_path

def remove_background(input_path, use_gpu=True):
    """Remove background from the selected image using GPU acceleration if available."""
    if not input_path:
        print("No file selected.")
        return

    print(f"Processing: {input_path}")

    try:
        # Open the input image
        input_image = Image.open(input_path)

        # Create a session with GPU support
        if use_gpu:
            # Configure CUDA execution provider with optimizations
            session = new_session(
                model_name="u2net",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print("Using GPU-accelerated processing...")
            output_image = remove(input_image, session=session)
        else:
            # Fallback to default (CPU)
            output_image = remove(input_image)

        # Generate output filename
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_no_bg.png"

        # Save the output image
        output_image.save(output_path)

        print(f"Background removed successfully!")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    print("Background Remover Tool (GPU Accelerated)")
    print("=" * 40)

    # Check GPU availability
    gpu_available = check_gpu_availability()
    print()

    # Select image file
    image_path = select_image()

    if image_path:
        # Remove background with GPU acceleration if available
        remove_background(image_path, use_gpu=gpu_available)
    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()
