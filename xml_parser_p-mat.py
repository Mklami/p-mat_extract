import re
import json
import os
from typing import Dict, List, Set, Tuple, Any
from collections import Counter

class DesignPatternParser:
    """Parser for extracting design patterns from P-MART XML files"""
    
    def __init__(self, input_file: str):
        """Initialize with the path to the XML file"""
        self.input_file = "/Users/mayasalami/Desktop/CS537_p-mat_parser/Design Pattern List v1.2.xml"
        self.content = ""
        self.projects = []
        self.pattern_types = set()
        self.stats = {
            "total_projects": 0,
            "total_patterns": 0,
            "total_architectures": 0,
            "total_classes": 0,
            "unique_classes": set()
        }
    
    def load_content(self) -> str:
        """Load and clean the XML content"""
        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Clean up some common escape sequences
        content = (content
            .replace('\\\\', '\\')  # Fix escaped backslashes
            .replace('\\"', '"')    # Fix escaped quotes
            .replace('\\<', '<')    # Fix escaped brackets
            .replace('\\>', '>')
            .replace('\\\n', '\n'))
        
        self.content = content
        return content
    
    def extract_projects(self) -> List[Dict]:
        """Extract all projects from the content"""
        projects = []
        program_regex = r'<program[^>]*>([\s\S]*?)<\/program>'
        for match in re.finditer(program_regex, self.content):
            program_content = match.group(0)
            
            # Extract program name
            name_match = re.search(r'<name\s*>(.*?)<\/name>', program_content)
            program_name = name_match.group(1) if name_match else "Unknown Program"
            
            # Extract program type
            type_match = re.search(r'<program type="([^"]*)"', program_content)
            program_type = type_match.group(1) if type_match else "Unknown"
            
            projects.append({
                "name": program_name,
                "type": program_type,
                "content": program_content,
                "patterns": []
            })
        
        self.projects = projects
        self.stats["total_projects"] = len(projects)
        return projects
    
    def extract_design_patterns(self) -> None:
        """Extract design patterns from each project"""
        pattern_regex = r'<designPattern name="([^"]*)">([\s\S]*?)<\/designPattern>'
        
        for project in self.projects:
            for match in re.finditer(pattern_regex, project["content"]):
                pattern_name = match.group(1)
                pattern_content = match.group(0)
                
                self.pattern_types.add(pattern_name)
                
                project["patterns"].append({
                    "name": pattern_name,
                    "content": pattern_content,
                    "microarchitectures": []
                })
                
                self.stats["total_patterns"] += 1
    
    def extract_microarchitectures(self) -> None:
        """Extract microarchitectures from each design pattern"""
        micro_regex = r'<microArchitecture number="(\d+)">([\s\S]*?)<\/microArchitecture>'
        
        for project in self.projects:
            for pattern in project["patterns"]:
                for match in re.finditer(micro_regex, pattern["content"]):
                    micro_number = match.group(1)
                    micro_content = match.group(0)
                    
                    # Extract roles and entities
                    roles_data = self._extract_roles(micro_content)
                    
                    # Extract any comments
                    comments = self._extract_comments(micro_content)
                    
                    pattern["microarchitectures"].append({
                        "number": micro_number,
                        "content": micro_content,
                        "roles": roles_data,
                        "comments": comments
                    })
                    
                    self.stats["total_architectures"] += 1
                    
                    # Add classes to the unique set
                    for role_type, entities in roles_data.items():
                        for entity in entities:
                            if entity.strip():
                                self.stats["unique_classes"].add(entity)
                                self.stats["total_classes"] += 1
    
    def _extract_roles(self, micro_content: str) -> Dict[str, List[str]]:
        """Extract roles and their entities from a microarchitecture"""
        roles_match = re.search(r'<roles>([\s\S]*?)<\/roles>', micro_content)
        if not roles_match:
            return {}
        
        roles_content = roles_match.group(1)
        roles_data = {}
        
        # Find all role types (clients, adapters, etc.)
        role_types_regex = r'<(\w+)s>([\s\S]*?)<\/\1s>'
        for role_match in re.finditer(role_types_regex, roles_content):
            role_type = role_match.group(1)
            role_content = role_match.group(2)
            
            # Extract entities for this role type
            entities = []
            entity_regex = r'<entity>([\s\S]*?)<\/entity>'
            for entity_match in re.finditer(entity_regex, role_content):
                entity_name = entity_match.group(1).strip()
                if entity_name:
                    entities.append(entity_name)
            
            roles_data[role_type] = entities
        
        return roles_data
    
    def _extract_comments(self, micro_content: str) -> str:
        """Extract any comments from a microarchitecture"""
        comment_match = re.search(r'<comments?>([\s\S]*?)<\/comments?>', micro_content)
        if comment_match:
            return comment_match.group(1).strip()
        return ""
    
    def parse(self) -> Dict:
        """Parse the entire file and return structured data"""
        self.load_content()
        self.extract_projects()
        self.extract_design_patterns()
        self.extract_microarchitectures()
        
        # Don't convert to length here - keep the set
        # self.stats["unique_classes"] = len(self.stats["unique_classes"])
        
        return {
            "projects": self.projects,
            "pattern_types": list(self.pattern_types),
            "stats": self.stats
        }
    
    # Update the save_to_json method in your file
    def save_to_json(self, output_file: str) -> None:
        """Save the parsed data to a JSON file"""
        data = self.parse()
        
        # Convert set to its size for JSON serialization
        if isinstance(data["stats"]["unique_classes"], set):
            data_copy = dict(data)
            stats_copy = dict(data_copy["stats"])
            stats_copy["unique_classes"] = len(data["stats"]["unique_classes"])
            data_copy["stats"] = stats_copy
            data = data_copy
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data saved to {output_file}")
    
    def generate_pattern_report(self, pattern_type: str) -> Dict:
        """Generate a report for a specific design pattern type across all projects"""
        report = {
            "pattern_type": pattern_type,
            "implementations": [],
            "total_implementations": 0,
            "projects_using": set(),
            "common_classes": {},
            "common_roles": {}
        }
        
        # Track class usage frequency
        class_counter = Counter()
        role_counter = Counter()
        
        for project in self.projects:
            for pattern in project["patterns"]:
                if pattern["name"] == pattern_type:
                    for micro in pattern["microarchitectures"]:
                        implementation = {
                            "project": project["name"],
                            "micro_id": micro["number"],
                            "roles": micro["roles"],
                            "comments": micro.get("comments", "")
                        }
                        report["implementations"].append(implementation)
                        report["total_implementations"] += 1
                        report["projects_using"].add(project["name"])
                        
                        # Count class occurrences
                        for role_type, entities in micro["roles"].items():
                            role_counter[role_type] += 1
                            for entity in entities:
                                class_counter[entity] += 1
        
        # Find common classes used in this pattern (appearing in multiple implementations)
        report["common_classes"] = {
            class_name: count for class_name, count in class_counter.items() 
            if count > 1 and class_name.strip()
        }
        
        # Find common roles used in this pattern
        report["common_roles"] = dict(role_counter)
        
        # Convert set to list for JSON serialization
        report["projects_using"] = list(report["projects_using"])
        
        return report
    
    def generate_project_report(self, project_name: str) -> Dict:
        """Generate a report for a specific project"""
        report = {
            "project_name": project_name,
            "patterns_used": [],
            "pattern_count": 0,
            "class_participation": {},
            "pattern_frequency": {}
        }
        
        # Find the project
        project = None
        for p in self.projects:
            if p["name"] == project_name:
                project = p
                break
        
        if not project:
            return {"error": f"Project {project_name} not found"}
        
        # Count pattern frequency
        pattern_counter = Counter()
        class_patterns = {}  # Track which patterns each class participates in
        
        for pattern in project["patterns"]:
            pattern_name = pattern["name"]
            pattern_counter[pattern_name] += 1
            
            for micro in pattern["microarchitectures"]:
                for role, entities in micro["roles"].items():
                    for entity in entities:
                        if entity not in class_patterns:
                            class_patterns[entity] = set()
                        class_patterns[entity].add(pattern_name)
        
        # Format the report
        report["patterns_used"] = list(pattern_counter.keys())
        report["pattern_count"] = len(pattern_counter)
        report["pattern_frequency"] = dict(pattern_counter)
        
        # Convert sets to lists for JSON
        for cls, patterns in class_patterns.items():
            report["class_participation"][cls] = list(patterns)
        
        return report
    
    def generate_llm_analysis_batch(self, output_dir: str, batch_size: int = 5) -> None:
        """Generate batch files for LLM analysis of design patterns"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Make sure we have data
        if not self.projects and not self.pattern_types:
            self.parse()
        
        # Process each pattern type
        for pattern_type in sorted(self.pattern_types):
            report = self.generate_pattern_report(pattern_type)
            
            # Skip if no implementations
            if not report["implementations"]:
                continue
            
            # Process implementations in batches
            for i in range(0, len(report["implementations"]), batch_size):
                batch = report["implementations"][i:i+batch_size]
                batch_num = i // batch_size + 1
                
                prompt = self._create_llm_prompt(pattern_type, batch)
                
                # Write the prompt to a file
                filename = f"{pattern_type.replace(' ', '_')}_{batch_num}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                print(f"Created LLM analysis file: {filename}")
    
    def _create_llm_prompt(self, pattern_type: str, implementations: List[Dict]) -> str:
        """Create an LLM prompt for analyzing design pattern implementations"""
        prompt = f"# Analysis of {pattern_type} Design Pattern Implementations\n\n"
        prompt += "Please analyze the following implementations of the"
        prompt += f" {pattern_type} design pattern across different projects.\n\n"
        prompt += "For each implementation:\n"
        prompt += "1. Evaluate how well it follows the design pattern principles\n"
        prompt += "2. Identify strengths and weaknesses\n"
        prompt += "3. Suggest refactoring opportunities\n"
        prompt += "4. Rate the implementation on a scale of 1-10\n\n"
        
        for i, impl in enumerate(implementations, 1):
            prompt += f"## Implementation {i}: {pattern_type} in {impl['project']} (ID: {impl['micro_id']})\n\n"
            
            # Add the roles and their classes
            prompt += "### Class Structure:\n"
            for role, entities in impl["roles"].items():
                if entities:
                    prompt += f"- **{role}**: {', '.join(entities)}\n"
            
            # Add any comments
            if impl.get("comments"):
                prompt += f"\n### Developer Comments:\n{impl['comments']}\n"
            
            prompt += "\n"
        
        # Add final instructions
        prompt += "## Overall Analysis\n\n"
        prompt += f"Based on these {len(implementations)} implementations of the {pattern_type} pattern:\n\n"
        prompt += "1. What are common implementation approaches?\n"
        prompt += "2. What are the best practices demonstrated?\n"
        prompt += "3. What are common pitfalls or anti-patterns?\n"
        prompt += "4. What would an ideal implementation look like?\n"
        prompt += "5. Provide a generic refactored example that addresses the common issues.\n"
        
        return prompt

    def find_class_pattern_participation(self, class_name: str) -> Dict:
        """Find all design patterns a specific class participates in"""
        result = {
            "class_name": class_name,
            "pattern_participation": [],
            "total_patterns": 0,
            "roles_played": Counter()
        }
        
        for project in self.projects:
            for pattern in project["patterns"]:
                for micro in pattern["microarchitectures"]:
                    for role, entities in micro["roles"].items():
                        if class_name in entities:
                            result["pattern_participation"].append({
                                "project": project["name"],
                                "pattern": pattern["name"],
                                "micro_id": micro["number"],
                                "role": role
                            })
                            result["roles_played"][role] += 1
        
        result["total_patterns"] = len(result["pattern_participation"])
        result["roles_played"] = dict(result["roles_played"])
        
        return result
    
    def generate_class_participation_report(self, output_file: str, min_patterns: int = 2) -> None:
        """Generate a report of classes that participate in multiple design patterns"""
        # Make sure we have data
        if not self.projects:
            self.parse()
        
        # Collect all classes and their pattern participation
        class_patterns = {}
        
        for project in self.projects:
            for pattern in project["patterns"]:
                pattern_name = pattern["name"]
                
                for micro in pattern["microarchitectures"]:
                    for role, entities in micro["roles"].items():
                        for entity in entities:
                            if entity not in class_patterns:
                                class_patterns[entity] = set()
                            class_patterns[entity].add(pattern_name)
        
        # Filter classes that participate in multiple patterns
        multi_pattern_classes = {
            cls: patterns 
            for cls, patterns in class_patterns.items() 
            if len(patterns) >= min_patterns and cls.strip()
        }
        
        # Sort by number of patterns (descending)
        sorted_classes = sorted(
            multi_pattern_classes.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Generate report
        report = {
            "multi_pattern_classes": [],
            "total_classes": len(sorted_classes)
        }
        
        for cls, patterns in sorted_classes:
            report["multi_pattern_classes"].append({
                "class_name": cls,
                "patterns": list(patterns),
                "pattern_count": len(patterns)
            })
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Class participation report saved to {output_file}")
        print(f"Found {len(sorted_classes)} classes that participate in {min_patterns}+ patterns")

if __name__ == "__main__":
    # Example usage
    parser = DesignPatternParser("p-mart-design-thing.xml")
    data = parser.parse()
    
    # Save the full data
    parser.save_to_json("design_patterns_data.json")
    
    # Generate pattern-specific reports
    os.makedirs("pattern_reports", exist_ok=True)
    for pattern_type in parser.pattern_types:
        report = parser.generate_pattern_report(pattern_type)
        with open(f"pattern_reports/{pattern_type.replace(' ', '_')}.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    # Generate project-specific reports
    os.makedirs("project_reports", exist_ok=True)
    for project in parser.projects:
        report = parser.generate_project_report(project["name"])
        with open(f"project_reports/{project['name'].replace(' ', '_')}.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    # Generate LLM analysis batch files
    parser.generate_llm_analysis_batch("llm_analysis_batches")
    
    # Generate class participation report
    parser.generate_class_participation_report("multi_pattern_classes.json")