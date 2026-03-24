import re

def slugify(txt):
    # CamelCase to snake_case transition
    text = re.sub(r'([A-Za-z0-9])([A-Z])', r'\1_\2', txt.strip())
    # Convert to uppercase
    text = text.upper()
    # Replace whitespace with underscore
    text = re.sub(r'\s+', '_', text)
    # Remove non-alphanumeric (excluding underscore)
    text = re.sub(r'[^A-Z0-9_]', '', text)
    # Collapse multiple underscores
    text = re.sub(r'_+', '_', text)
    return text.strip('_')

print(f"'ExpectAFight' -> '{slugify('ExpectAFight')}'")
print(f"'ExpectAfight' -> '{slugify('ExpectAfight')}'")
