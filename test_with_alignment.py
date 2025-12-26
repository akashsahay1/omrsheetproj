"""Test OMR processing with FeatureBasedAlignment enabled."""
import os
import sys

# Add OMRChecker to path
sys.path.insert(0, str("F:\\Projects\\omrsheetproj\\OMRChecker"))

from pathlib import Path
import shutil
import tempfile

def test_processing():
    """Test processing a few images with the updated template."""

    # Create temp processing directory
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Copy template files
    template_src = Path("F:\\Projects\\omrsheetproj\\OMRChecker\\inputs\\VidyaBharti2025")
    shutil.copy(template_src / "template.json", input_dir / "template.json")
    shutil.copy(template_src / "omr-filled.png", input_dir / "omr-filled.png")

    # Copy a few test images
    test_images = ["41_624.jpg", "41_625.jpg", "41_630.jpg", "41_635.jpg", "41_636.jpg"]
    for img in test_images:
        src = Path("filledomrsheet") / img
        if src.exists():
            shutil.copy(src, input_dir / img)

    print(f"Processing {len(test_images)} images with FeatureBasedAlignment...")
    print("="*60)

    try:
        # Import and run OMRChecker
        from src.entry import entry_point

        args = {
            'debug': False,
            'output_dir': str(output_dir),
            'autoAlign': False,
            'setLayout': False
        }

        entry_point(input_dir, args)

        # Read results
        results_file = output_dir / "Results" / "Results_1.csv"
        if results_file.exists():
            print(f"\nResults from {results_file}:")
            with open(results_file, 'r') as f:
                print(f.read())
        else:
            # Check for any CSV file
            for csv_file in (output_dir / "Results").glob("*.csv"):
                print(f"\nResults from {csv_file}:")
                with open(csv_file, 'r') as f:
                    print(f.read())

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_processing()
