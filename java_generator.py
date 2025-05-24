import xml.etree.ElementTree as ET
from typing import Dict, List, Any

class JavaCodeGenerator:
    def __init__(self, scxml_content: str):
        self.scxml_content = scxml_content
        self.root = ET.fromstring(scxml_content)
        self.classes = {}
        self.relationships = []
        self._parse_scxml()

    def generate(self) -> str:
        """Generate Java code from the SCXML content."""
        java_code = []
        
        # Add package and imports
        java_code.extend([
            'package generated;',
            '',
            'import java.util.*;',
            'import java.io.*;',
            '',
        ])
        
        # Generate each class
        for class_name, class_data in self.classes.items():
            java_code.extend(self._generate_class(class_name, class_data))
        
        return '\n'.join(java_code)

    def _parse_scxml(self):
        """Parse the SCXML content and extract classes and relationships."""
        # Extract classes and their attributes
        for state in self.root.findall('.//state'):
            class_name = state.get('id').capitalize()
            self.classes[class_name] = {
                'attributes': [],
                'methods': [],
                'relationships': []
            }
            
            # Extract attributes from onentry logs
            for onentry in state.findall('onentry'):
                for log in onentry.findall('log'):
                    expr = log.get('expr')
                    if expr and expr.startswith("'Entering"):
                        # Parse attributes from the log message
                        attrs = expr.split('state')[0].split('Entering')[1].strip()
                        if attrs:
                            self.classes[class_name]['attributes'].extend(
                                [attr.strip() for attr in attrs.split(',')]
                            )
            
            # Extract relationships from transitions
            for transition in state.findall('transition'):
                target = transition.get('target').capitalize()
                if target != class_name:
                    self.classes[class_name]['relationships'].append(target)

    def _generate_class(self, class_name: str, class_data: Dict) -> List[str]:
        """Generate Java code for a single class."""
        class_lines = [
            f'public class {class_name} {{',
            '    // Attributes'
        ]
        
        # Add attributes
        for attr in class_data['attributes']:
            if attr:
                # Default to String type if no type is specified
                attr_type = 'String'
                attr_name = attr
                if ':' in attr:
                    attr_type, attr_name = attr.split(':')
                    attr_type = attr_type.strip()
                    attr_name = attr_name.strip()
                
                class_lines.append(f'    private {attr_type} {attr_name};')
        
        # Add constructors
        class_lines.extend([
            '',
            '    // Constructors',
            f'    public {class_name}() {{',
            '    }',
            ''
        ])
        
        # Add getters and setters
        class_lines.extend([
            '    // Getters and Setters'
        ])
        
        for attr in class_data['attributes']:
            if attr:
                attr_type = 'String'
                attr_name = attr
                if ':' in attr:
                    attr_type, attr_name = attr.split(':')
                    attr_type = attr_type.strip()
                    attr_name = attr_name.strip()
                
                # Getter
                class_lines.extend([
                    f'    public {attr_type} get{attr_name.capitalize()}() {{',
                    f'        return {attr_name};',
                    '    }',
                    ''
                ])
                
                # Setter
                class_lines.extend([
                    f'    public void set{attr_name.capitalize()}({attr_type} {attr_name}) {{',
                    f'        this.{attr_name} = {attr_name};',
                    '    }',
                    ''
                ])
        
        # Add relationship methods
        if class_data['relationships']:
            class_lines.extend([
                '    // Relationship methods'
            ])
            
            for rel in class_data['relationships']:
                # Add relationship field
                class_lines.extend([
                    f'    private {rel} {rel.lower()};',
                    ''
                ])
                
                # Add getter and setter for relationship
                class_lines.extend([
                    f'    public {rel} get{rel}() {{',
                    f'        return {rel.lower()};',
                    '    }',
                    '',
                    f'    public void set{rel}({rel} {rel.lower()}) {{',
                    f'        this.{rel.lower()} = {rel.lower()};',
                    '    }',
                    ''
                ])
        
        # Add toString method
        class_lines.extend([
            '    @Override',
            '    public String toString() {',
            '        return "' + class_name + '{" +',
            '            ' + ', '.join(f'"{attr.split(":")[0].strip()}=\'" + {attr.split(":")[0].strip()} + "\'"' for attr in class_data['attributes'] if attr) + ' +',
            '            "}"',
            '    }',
            '}',
            ''
        ])
        
        return class_lines 