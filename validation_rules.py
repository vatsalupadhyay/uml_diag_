import re

# Regex patterns for checking attributes and methods
ATTRIBUTE_PATTERN = r'[-+]?\s*\w+\s*:\s*\w+(?:\[\])?'  # Handles arrays and optional visibility
METHOD_PATTERN = r'[-+]?\s*\w+\s*\([^)]*\)(?:\s*:\s*\w+)?'  # More flexible method pattern

def validate_class_diagram(extracted_texts):
    errors = []
    if not extracted_texts:
        return ["No text blocks detected in the UML diagram."]
    
    for text_block in extracted_texts:
        lines = text_block.split("\n")
        if not lines:
            continue
            
        # Extract class name (more flexible)
        class_name = None
        for line in lines:
            # Skip common OCR artifacts and headers
            if line.strip() and not line.lower().startswith(('class:', 'attributes:', 'methods:', 'block')):
                class_name = line.strip()
                break
        
        if not class_name:
            continue  # Skip if no class name found
            
        # Check for attributes and methods with more flexible patterns
        attributes = []
        methods = []
        
        for line in lines[1:]:  # Skip the class name line
            line = line.strip()
            if not line:
                continue
                
            # Clean up common OCR artifacts
            line = line.replace('|', 'I').replace('[]', '').replace('{}', '()')
            
            if '(' in line and ')' in line:
                if re.search(METHOD_PATTERN, line):
                    methods.append(line)
            elif ':' in line:
                if re.search(ATTRIBUTE_PATTERN, line):
                    attributes.append(line)
        
        # Only add errors if we found a valid class but missing components
        if class_name and not attributes and not methods:
            errors.append(f"Class {class_name} appears to be missing both attributes and methods")
        elif class_name and not attributes:
            errors.append(f"Class {class_name} appears to be missing attributes")
        elif class_name and not methods:
            errors.append(f"Class {class_name} appears to be missing methods")
    
    return errors if errors else ["Valid UML class diagram."]
