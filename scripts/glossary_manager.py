#!/usr/bin/env python3
"""
Glossary Management System for Legal Documentation
Maintains and updates technical concept explanations
"""

import os
import json
from datetime import datetime
from pathlib import Path
import re

class GlossaryManager:
    """Manage technical glossary for legal documentation"""
    
    def __init__(self, glossary_path="/home/cy/Legal/Glossary"):
        self.glossary_path = Path(glossary_path)
        self.concepts_dir = self.glossary_path / "concepts"
        self.concepts_dir.mkdir(parents=True, exist_ok=True)
        self.index = self.load_index()
    
    def load_index(self):
        """Load or create glossary index"""
        index_file = self.glossary_path / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_index(self):
        """Save glossary index"""
        index_file = self.glossary_path / "index.json"
        with open(index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def add_concept(self, term, one_liner, explanation, analogy, 
                    significance=None, related_terms=None, category="General"):
        """Add a new concept to the glossary"""
        
        # Generate filename
        safe_filename = re.sub(r'[^a-z0-9_]', '_', term.lower())
        
        # Find next available number
        existing_files = sorted(self.concepts_dir.glob("*.md"))
        if existing_files:
            last_num = int(existing_files[-1].stem.split('_')[0])
            file_num = last_num + 1
        else:
            file_num = 1
        
        filename = f"{file_num:02d}_{safe_filename}.md"
        filepath = self.concepts_dir / filename
        
        # Create concept entry
        content = f"""# {term}

**One-Sentence Summary**: {one_liner}

---

## Plain English Explanation

{explanation}

## Real-World Analogy

{analogy}
"""
        
        if significance:
            content += f"""
## Why It Matters

{significance}
"""
        
        if related_terms:
            content += f"""
## Related Concepts
"""
            for related in related_terms:
                content += f"- [{related}]({self.find_concept_file(related)})\n"
        
        content += f"""
---
*Version: 1.0 | Last Updated: {datetime.now().strftime('%B %d, %Y')}*"""
        
        # Write file
        with open(filepath, 'w') as f:
            f.write(content)
        
        # Update index
        self.index[term] = {
            'file': filename,
            'category': category,
            'one_liner': one_liner,
            'created': datetime.now().isoformat(),
            'version': '1.0'
        }
        self.save_index()
        
        print(f"✓ Added concept: {term} -> {filename}")
        return filepath
    
    def find_concept_file(self, term):
        """Find the file for a given concept"""
        if term in self.index:
            return self.index[term]['file']
        
        # Try to find by searching files
        for file in self.concepts_dir.glob("*.md"):
            with open(file, 'r') as f:
                first_line = f.readline()
                if term.lower() in first_line.lower():
                    return file.name
        
        return "#"  # Return anchor if not found
    
    def update_concept(self, term, updates):
        """Update an existing concept"""
        if term not in self.index:
            print(f"Concept '{term}' not found")
            return
        
        filepath = self.concepts_dir / self.index[term]['file']
        
        # Read current content
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Update version
        old_version = self.index[term]['version']
        new_version = f"{float(old_version) + 0.1:.1f}"
        
        # Update content (this is simplified - in practice would be more sophisticated)
        for key, value in updates.items():
            if f"## {key}" in content:
                # Replace section
                pattern = f"## {key}.*?(?=##|---)"
                replacement = f"## {key}\n\n{value}\n\n"
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Update version and date
        content = re.sub(
            r'Version: [\d.]+',
            f'Version: {new_version}',
            content
        )
        content = re.sub(
            r'Last Updated: .*?\*',
            f'Last Updated: {datetime.now().strftime("%B %d, %Y")}*',
            content
        )
        
        # Write updated content
        with open(filepath, 'w') as f:
            f.write(content)
        
        # Update index
        self.index[term]['version'] = new_version
        self.index[term]['updated'] = datetime.now().isoformat()
        self.save_index()
        
        print(f"✓ Updated concept: {term} (v{new_version})")
    
    def generate_legal_brief_glossary(self, terms_used):
        """Generate a glossary appendix for a legal brief"""
        
        glossary = "# Glossary of Technical Terms\n\n"
        
        for term in sorted(terms_used):
            if term in self.index:
                info = self.index[term]
                glossary += f"**{term}**: {info['one_liner']}\n"
                glossary += f"*See: {info['file']} for full explanation*\n\n"
        
        return glossary
    
    def check_document_terms(self, document_path):
        """Scan a document and identify technical terms that need glossary entries"""
        
        # List of technical terms to look for
        technical_terms = list(self.index.keys())
        
        # Add common technical terms that might not be in glossary yet
        additional_terms = [
            "neural network", "machine learning", "artificial intelligence",
            "training", "model", "algorithm", "optimization", "gradient",
            "loss function", "backpropagation", "tensor", "matrix"
        ]
        
        all_terms = technical_terms + additional_terms
        
        with open(document_path, 'r') as f:
            content = f.read().lower()
        
        found_terms = []
        missing_terms = []
        
        for term in all_terms:
            if term.lower() in content:
                if term in self.index:
                    found_terms.append(term)
                else:
                    missing_terms.append(term)
        
        return {
            'found': found_terms,
            'missing': missing_terms
        }
    
    def export_for_legal(self, output_path=None):
        """Export glossary in legal-friendly format"""
        
        if not output_path:
            output_path = self.glossary_path / "legal_glossary.pdf"
        
        # For now, create a markdown version
        # In production, would convert to PDF
        
        content = """# Technical Glossary for Legal Professionals
## 2π Variance Regulation Patent

---

## Disclaimer
This glossary provides simplified explanations of technical concepts for legal professionals. 
These explanations are intended for educational purposes and should not be considered 
definitive technical definitions.

---

## Terms

"""
        
        # Sort terms alphabetically
        for term in sorted(self.index.keys()):
            info = self.index[term]
            filepath = self.concepts_dir / info['file']
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    concept_content = f.read()
                    # Extract just the summary and explanation
                    lines = concept_content.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('**One-Sentence Summary**'):
                            summary = line.replace('**One-Sentence Summary**: ', '')
                            content += f"### {term}\n{summary}\n\n"
                            break
        
        # Save
        output_md = output_path.with_suffix('.md')
        with open(output_md, 'w') as f:
            f.write(content)
        
        print(f"✓ Exported glossary to {output_md}")
        return output_md


# Quick-add functions for common concepts
def add_ml_concept(term, simple_explanation, why_it_matters):
    """Quick function to add machine learning concepts"""
    manager = GlossaryManager()
    return manager.add_concept(
        term=term,
        one_liner=simple_explanation,
        explanation=why_it_matters,
        analogy="[Add appropriate analogy]",
        category="Machine Learning"
    )

def add_measurement(term, what_it_measures, why_we_care):
    """Quick function to add measurement/metric concepts"""
    manager = GlossaryManager()
    return manager.add_concept(
        term=term,
        one_liner=what_it_measures,
        explanation=why_we_care,
        analogy="[Add appropriate analogy]",
        category="Measurements"
    )


if __name__ == "__main__":
    # Example usage
    manager = GlossaryManager()
    
    # Add a new concept
    manager.add_concept(
        term="Purple Line Events",
        one_liner="Warning signals that occur when an AI system approaches instability, like a smoke alarm before a fire",
        explanation="""Purple Line Events are specific moments during AI training when the system 
violates the 2π speed limit. We call them 'Purple Line' because in our monitoring system, 
these violations show up as purple markers on the graph.

Think of them like warning lights on your car's dashboard - they tell you something needs 
attention before it becomes a serious problem.""",
        analogy="""It's like the rumble strips on the side of a highway - they warn you that you're 
drifting out of your lane before you actually leave the road. Purple Line Events warn us that 
the AI is drifting toward instability before it actually fails.""",
        significance="These events are critical early warning indicators that help us prevent AI training failures",
        related_terms=["2π Regulation Principle", "Compliance Rate", "Training Stability"],
        category="Monitoring"
    )
    
    # Check a document for terms
    # results = manager.check_document_terms("/home/cy/Legal/Patent_2Pi/patent_application.md")
    # print(f"Found terms: {results['found']}")
    # print(f"Missing glossary entries: {results['missing']}")
    
    # Export for legal use
    manager.export_for_legal()