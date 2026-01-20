import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os

def select_image():
    """Open file dialog to select an image file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select an image to flip vertically",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("BMP files", "*.bmp"),
            ("GIF files", "*.gif"),
            ("TIFF files", "*.tiff"),
            ("All files", "*.*")
        ]
    )

    return file_path

def flip_image_vertically(input_path):
    """Flip the selected image vertically (upside down)."""
    if not input_path:
        print("No file selected.")
        return

    print(f"Processing: {input_path}")

    try:
        # Open the input image
        input_image = Image.open(input_path)

        # Flip the image vertically
        flipped_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)

        # Generate output filename
        base_name = os.path.splitext(input_path)[0]
        extension = os.path.splitext(input_path)[1]
        output_path = f"{base_name}_vertical_flip{extension}"

        # Save the flipped image
        flipped_image.save(output_path)

        print(f"Image flipped vertically successfully!")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    print("Vertical Image Flip Tool")
    print("=" * 40)

    # Select image file
    image_path = select_image()

    if image_path:
        # Flip image vertically
        flip_image_vertically(image_path)
    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()
