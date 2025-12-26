"""Test script for name cleaning function."""
import re

def clean_detected_name(raw_name):
    if not raw_name:
        return raw_name
    name = raw_name

    # Step 1: Remove trailing W's
    name = re.sub(r'\s*W+\s*$', '', name)

    # Step 2: Remove W attached to end of words
    name = re.sub(r'([A-VX-Z])W(\s|$)', r'\1\2', name)

    # Step 3: Remove multiple W's and M's
    name = re.sub(r'W{2,}', '', name)
    name = re.sub(r'M{2,}', 'M', name)

    # Step 4: Remove triple+ letters
    def remove_triple_plus(match):
        char = match.group(1)
        return char * 2
    name = re.sub(r'([A-LN-VX-Z])\1{2,}', remove_triple_plus, name)

    # Step 5: Remove WM/MW noise
    name = re.sub(r'\s+[WM]{2,}', '', name)

    # Step 6: Remove isolated char after spaces
    name = re.sub(r'\s{2,}[A-Z]$', '', name)

    # Step 7: Clean excessive spaces
    name = re.sub(r'\s{3,}', '  ', name)

    # Step 8: Clean trailing J/K/W/M
    name = re.sub(r'\s+[JKWM]$', '', name)

    # Final
    name = name.strip()
    while re.search(r'\s{2,}[A-Z]$', name):
        name = re.sub(r'\s{2,}[A-Z]$', '', name).strip()
    return name

# Test cases from logs
test_names = [
    'AARI KUMARI             W',
    'SHIVANH KUMARW          W',
    'KOKALSINGH  WWMM        W',
    'CHANDANI KMARI   WWW',
    'MILANKKMAR DAS MM       W',
    'SUROJIT SIGH     WWWW',
    'RIYANSH KM R            W',
    'KARAN M TO              W',
    'PRIY NSU KUMAR MANDAL   K',
    'ROSG  R JWAR            K',
]

print("Name Cleaning Test Results:")
print("-" * 70)
for name in test_names:
    cleaned = clean_detected_name(name)
    print(f"'{name}'")
    print(f"  -> '{cleaned}'")
    print()
