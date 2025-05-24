import re
from typing import List, Dict, Any

class UMLProcessor:
    def __init__(self, text: str):
        self.text = text
        self.classes = []
        self.attributes = {}
        self.relationships = []
        self._extract_classes()

    def validate(self) -> List[str]:
        """Validate the UML diagram."""
        results = []
        
        # Check if we found any classes
        if not self.classes:
            results.append("No classes found in the UML diagram")
        else:
            results.append(f"Found {len(self.classes)} classes: {', '.join(self.classes)}")
        
        # Check each class
        for class_name in self.classes:
            # Check if class has attributes
            if class_name not in self.attributes or not self.attributes[class_name]:
                results.append(f"Warning: Class {class_name} has no attributes")
            else:
                results.append(f"Class {class_name} has {len(self.attributes[class_name])} attributes")
                # Validate attribute types
                for attr in self.attributes[class_name]:
                    if not self._is_valid_type(attr['type']):
                        results.append(f"Warning: Invalid type '{attr['type']}' for attribute '{attr['name']}' in class {class_name}")
        
        # Check relationships
        if not self.relationships:
            results.append("No relationships found between classes")
        else:
            results.append(f"Found {len(self.relationships)} relationships between classes")
            # Validate relationships
            for rel in self.relationships:
                if rel['source'] not in self.classes:
                    results.append(f"Warning: Source class '{rel['source']}' in relationship not found")
                if rel['target'] not in self.classes:
                    results.append(f"Warning: Target class '{rel['target']}' in relationship not found")
        
        return results

    def _is_valid_type(self, type_name: str) -> bool:
        """Check if a type is valid."""
        valid_types = {
            'String', 'int', 'long', 'float', 'double', 'boolean', 'char',
            'Integer', 'Long', 'Float', 'Double', 'Boolean', 'Character',
            'Date', 'DateTime', 'List', 'Set', 'Map', 'Collection'
        }
        return type_name in valid_types

    def generate_scxml(self) -> str:
        """Generate SCXML representation of the UML diagram."""
        scxml = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<scxml xmlns="http://www.w3.org/2005/07/scxml" version="1.0"',
                 '       initial="initial">']
        
        # Add states for each class
        for class_name in self.classes:
            # Get attributes for this class
            attrs = []
            if class_name in self.attributes:
                attrs = [f"{attr['name']}:{attr['type']}" for attr in self.attributes[class_name]]
            
            scxml.extend([
                f'    <state id="{class_name}" initial="initial">',
                '        <onentry>',
                f'            <log expr="\'Entering {class_name} state with attributes: {", ".join(attrs)}\'" />',
                '        </onentry>',
                '        <onexit>',
                f'            <log expr="\'Exiting {class_name} state\'" />',
                '        </onexit>'
            ])
            
            # Add transitions based on relationships
            for rel in self.relationships:
                if rel['source'] == class_name:
                    scxml.extend([
                        f'        <transition event="to_{rel["target"]}" target="{rel["target"]}" />'
                    ])
            
            scxml.append('    </state>')
        
        # Add transitions between states
        for rel in self.relationships:
            scxml.extend([
                f'    <transition event="connect_{rel["source"]}_{rel["target"]}"',
                f'              source="{rel["source"]}"',
                f'              target="{rel["target"]}" />'
            ])
        
        scxml.extend([
            '    <state id="initial">',
            '        <transition event="start" target="initial" />',
            '    </state>',
            '</scxml>'
        ])
        
        return '\n'.join(scxml)

    def _extract_classes(self):
        """Extract class names from the text."""
        # Look for class definitions in the format: class ClassName or ClassName
        class_pattern = r'(?:class\s+)?([A-Z][a-zA-Z0-9_]*)\s*{'
        self.classes = re.findall(class_pattern, self.text)
        
        # Remove duplicates while preserving order
        self.classes = list(dict.fromkeys(self.classes))
        
        # Extract attributes and methods for each class
        for class_name in self.classes:
            class_block = self._extract_class_block(class_name)
            self._extract_attributes(class_name, class_block)
            self._extract_methods(class_name, class_block)

    def _extract_class_block(self, class_name: str) -> str:
        """Extract the block of text containing a class definition."""
        # Look for text between class name and next class or end of text
        pattern = f'{class_name}\\s*{{(.*?)}}(?=\\s*(?:class\\s+[A-Z]|[A-Z][a-zA-Z0-9_]*\\s*{{|$))'
        match = re.search(pattern, self.text, re.DOTALL)
        return match.group(1) if match else ""

    def _extract_attributes(self, class_name: str, class_block: str):
        """Extract attributes from a class block."""
        # Look for attributes in the format: + attributeName: type or - attributeName: type
        attr_pattern = r'[+-]\s*([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_]+)'
        matches = re.finditer(attr_pattern, class_block)
        
        self.attributes[class_name] = []
        for match in matches:
            attr_name = match.group(1)
            attr_type = match.group(2)
            
            # Clean up common OCR mistakes in attribute types
            attr_type = attr_type.replace('Sting', 'String')
            attr_type = attr_type.replace('Int', 'int')
            attr_type = attr_type.replace('Float', 'float')
            attr_type = attr_type.replace('Bool', 'boolean')
            attr_type = attr_type.replace('Nurnber', 'Number')
            attr_type = attr_type.replace('Dater', 'Date')
            attr_type = attr_type.replace('Ion', 'Long')
            attr_type = attr_type.replace('Lat', 'Long')
            attr_type = attr_type.replace('Ant', 'int')
            attr_type = attr_type.replace('Pice', 'Price')
            attr_type = attr_type.replace('No', 'int')
            attr_type = attr_type.replace('L', 'Long')
            
            self.attributes[class_name].append({
                'name': attr_name,
                'type': attr_type
            })

    def _extract_methods(self, class_name: str, class_block: str):
        """Extract methods from a class block."""
        # Look for methods in the format: + methodName(): returnType or - methodName(): returnType
        method_pattern = r'[+-]\s*([a-zA-Z0-9_]+)\s*\(\s*\)\s*:\s*([a-zA-Z0-9_]+)'
        matches = re.finditer(method_pattern, class_block)
        
        if class_name not in self.attributes:
            self.attributes[class_name] = []
        
        for match in matches:
            method_name = match.group(1)
            return_type = match.group(2)
            
            # Clean up common OCR mistakes in return types
            return_type = return_type.replace('Sting', 'String')
            return_type = return_type.replace('Int', 'int')
            return_type = return_type.replace('Float', 'float')
            return_type = return_type.replace('Bool', 'boolean')
            return_type = return_type.replace('Nurnber', 'Number')
            return_type = return_type.replace('Dater', 'Date')
            return_type = return_type.replace('Ion', 'Long')
            return_type = return_type.replace('Lat', 'Long')
            return_type = return_type.replace('Ant', 'int')
            return_type = return_type.replace('Pice', 'Price')
            return_type = return_type.replace('No', 'int')
            return_type = return_type.replace('L', 'Long')
            
            self.attributes[class_name].append({
                'name': method_name,
                'type': return_type
            })

    def _extract_relationships(self):
        """Extract relationships between classes."""
        # Look for relationships in the format: Class1 -> Class2
        rel_pattern = r'([A-Z][a-zA-Z0-9_]*)\s*->\s*([A-Z][a-zA-Z0-9_]*)'
        matches = re.finditer(rel_pattern, self.text)
        
        for match in matches:
            self.relationships.append({
                'source': match.group(1),
                'target': match.group(2)
            }) 