import fitz  # PyMuPDF
import argparse
import sys
from pathlib import Path


def select_file_dialog(title="Select a file", filetypes=None):
    """Open a Windows file selection dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        if filetypes is None:
            filetypes = [("PDF files", "*.pdf"), ("All files", "*.*")]

        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )

        root.destroy()
        return file_path if file_path else None

    except ImportError:
        print("Error: tkinter is not available for file dialog.")
        return None


def select_folder_dialog(title="Select output folder"):
    """Open a Windows folder selection dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        folder_path = filedialog.askdirectory(title=title)

        root.destroy()
        return folder_path if folder_path else None

    except ImportError:
        print("Error: tkinter is not available for file dialog.")
        return None


def convert_pdf_to_png(input_pdf, output_folder, dpi=150, prefix=None):
    """
    Convert each page of a PDF to a PNG image.

    Args:
        input_pdf: Path to the input PDF file
        output_folder: Directory to save PNG files
        dpi: Resolution for output images (default: 150)
        prefix: Optional prefix for output filenames
    """
    input_path = Path(input_pdf)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_pdf}")
        sys.exit(1)

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = input_path.stem

    try:
        doc = fitz.open(input_pdf)
    except Exception as e:
        print(f"Error: Could not open PDF file: {e}")
        sys.exit(1)

    # Calculate zoom factor from DPI (72 is the base PDF DPI)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    print(f"Converting {len(doc)} page(s) at {dpi} DPI...\n")

    output_files = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if len(doc) == 1:
            output_file = output_path / f"{prefix}.png"
        else:
            output_file = output_path / f"{prefix}_page_{page_num + 1:03d}.png"

        pix.save(str(output_file))
        output_files.append(output_file)
        print(f"Saved: {output_file.name}")

    doc.close()

    print(f"\nCompleted! {len(output_files)} image(s) saved to: {output_folder}")
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to PNG images."
    )
    parser.add_argument(
        "-i", "--input",
        help="Input PDF file path (opens file dialog if not specified)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output folder path (opens folder dialog if not specified)"
    )
    parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=150,
        help="Output resolution in DPI (default: 150)"
    )
    parser.add_argument(
        "-p", "--prefix",
        help="Prefix for output filenames (default: input filename)"
    )

    args = parser.parse_args()

    # Get input file
    input_file = args.input
    if not input_file:
        print("Please select the input PDF file...")
        input_file = select_file_dialog(title="Select PDF to convert")
        if not input_file:
            print("No input file selected. Exiting.")
            sys.exit(1)

    # Get output folder
    output_folder = args.output
    if not output_folder:
        print("Please select the output folder...")
        output_folder = select_folder_dialog(title="Select folder for PNG output")
        if not output_folder:
            print("No output folder selected. Exiting.")
            sys.exit(1)

    print(f"\nInput:  {input_file}")
    print(f"Output: {output_folder}")
    print(f"DPI:    {args.dpi}\n")

    convert_pdf_to_png(input_file, output_folder, dpi=args.dpi, prefix=args.prefix)


if __name__ == "__main__":
    main()
