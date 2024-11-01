import re

# Regex patterns for checking attributes and methods
ATTRIBUTE_PATTERN = r'\b\w+\s*:\s*\w+\b'  # Example: attribute_name: int
METHOD_PATTERN = r'\b\w+\s*\(.*\)\s*:\s*\w+\b'  # Example: method_name(param: int): void

def validate_class_diagram(shapes, texts):
    errors = []
    if not shapes or not texts:
        return ["No shapes or text detected in the UML diagram."]
    
    for i, text in enumerate(texts):
        if not text.strip():
            errors.append(f"Class {i + 1} is missing a name.")
            continue

        # Extract different sections from the class text (assuming OCR output contains class name, attributes, and methods)
        class_sections = text.split("\n")
        class_name = class_sections[0]
        attributes = [line for line in class_sections if re.search(ATTRIBUTE_PATTERN, line)]
        methods = [line for line in class_sections if re.search(METHOD_PATTERN, line)]

        # Rule 1: Class Naming
        if not class_name or len(class_name.split()) < 2:
            errors.append(f"Class {i + 1} does not have a meaningful name.")
        
        # Rule 2: Attributes (Ensure that each class has attributes with correct data types)
        if not attributes:
            errors.append(f"Class {i + 1} is missing attributes or they are not properly defined with data types.")
        
        # Rule 3: Methods (Ensure that methods have return types and parameters)
        if not methods:
            errors.append(f"Class {i + 1} is missing methods or they are not properly defined.")
        
        # Rule 4: Visibility (Ensure visibility symbols like +, -, # are present)
        if not any(symbol in text for symbol in ['+', '-', '#']):
            errors.append(f"Class {i + 1} is missing visibility symbols (public, private, or protected).")
        
        # Rule 5: Inheritance (Check for inheritance arrows or proper notation)
        # Note: Inheritance check would require graphical analysis, but we can assume class relationships are given as text.

        # Rule 6: Associations, Aggregation, Composition, Multiplicity
        # These would generally be checked graphically (lines and arrows between classes), so skip for now unless you integrate shape analysis.

    return errors if errors else ["Valid UML class diagram."]
