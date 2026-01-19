import tkinter as tk
from tkinter import filedialog
from rembg import remove
from PIL import Image
import os

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

def remove_background(input_path):
    """Remove background from the selected image."""
    if not input_path:
        print("No file selected.")
        return

    print(f"Processing: {input_path}")

    try:
        # Open the input image
        input_image = Image.open(input_path)

        # Remove the background
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
    print("Background Remover Tool")
    print("=" * 40)

    # Select image file
    image_path = select_image()

    if image_path:
        # Remove background
        remove_background(image_path)
    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()
