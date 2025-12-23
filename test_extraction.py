"""Test full extraction after rotation fix"""
import sys
sys.path.insert(0, '.')
from src.core.omr_engine import OMREngine

engine = OMREngine()
result = engine.process_image('omrfilled/omr-filled.png')

print('='*60)
print('EXTRACTION RESULTS')
print('='*60)
if result.candidate:
    c = result.candidate
    print(f'Name: {c.name}')
    print(f'Roll Number: {c.roll_number}')
    print(f'Class: {c.class_}')
    print(f'Section: {c.section}')
    print(f'School Code: {c.school_code}')
else:
    print('No candidate extracted')
    print(f'Errors: {result.errors}')
