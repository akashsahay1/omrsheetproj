"""
Test single OMR image processing
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.core.omr_engine import OMREngine
from src.core.template import TemplateLoader

def main():
    print("=" * 60)
    print("OMR Single Image Test")
    print("=" * 60)

    # Check template loading
    print("\n1. Testing template loading...")
    loader = TemplateLoader()
    print(f"   Templates dir: {loader.templates_dir}")
    print(f"   Templates dir exists: {loader.templates_dir.exists()}")

    template = loader.load_template("vidya_bharti_2025")
    if template:
        print(f"   Template loaded: {template.name}")
        print(f"   Page dimensions: {template.page_dimensions}")
        print(f"   Bubble dimensions: {template.bubble_dimensions}")
        print(f"   Field blocks: {list(template.field_blocks.keys())}")
    else:
        print("   ERROR: Template not loaded!")
        return

    # Test OMR engine
    print("\n2. Testing OMR engine...")
    engine = OMREngine()
    print(f"   Engine template: {engine.template.name}")

    # Process filled sample
    sample_path = project_root / "omrfilled" / "omr-filled.png"
    print(f"\n3. Processing: {sample_path}")

    if not sample_path.exists():
        print(f"   ERROR: Sample file not found!")
        return

    result = engine.process_image(str(sample_path))

    print(f"\n4. Results:")
    print(f"   Success: {result.success}")

    if result.success and result.candidate:
        c = result.candidate
        print(f"   Name: '{c.name}'")
        print(f"   Roll Number: '{c.roll_number}'")
        print(f"   Class: '{c.class_}'")
        print(f"   Section: '{c.section}'")
        print(f"   School Code: '{c.school_code}'")
        print(f"   Answered: {c.answered_count}/60")
        print(f"   Multi-marked: {c.multi_marked_count}")
        print(f"\n   First 10 answers:")
        for i in range(1, 11):
            print(f"     Q{i}: {c.get_answer(i)}")
    else:
        print(f"   Error: {result.error_message}")

    if result.warnings:
        print(f"\n   Warnings:")
        for w in result.warnings:
            print(f"     - {w}")

    print(f"\n   Processing time: {result.processing_time_ms:.1f} ms")


if __name__ == "__main__":
    main()
