#!/usr/bin/env python3
"""
Validation script for Challenge 1B
Tests the persona-driven document analysis solution
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys

def validate_output_format(output_file):
    """Validate the JSON output format for Challenge 1B"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required top-level fields
        required_fields = ['metadata', 'extracted_sections', 'subsection_analysis']
        for field in required_fields:
            if field not in data:
                return False, f"Missing '{field}' field"
        
        # Validate metadata
        metadata = data['metadata']
        metadata_fields = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
        for field in metadata_fields:
            if field not in metadata:
                return False, f"Missing metadata.{field}"
        
        # Validate extracted_sections
        sections = data['extracted_sections']
        if not isinstance(sections, list):
            return False, "'extracted_sections' must be a list"
        
        for i, section in enumerate(sections):
            section_fields = ['document', 'section_title', 'importance_rank', 'page_number']
            for field in section_fields:
                if field not in section:
                    return False, f"extracted_sections[{i}] missing '{field}'"
        
        # Validate subsection_analysis
        subsections = data['subsection_analysis']
        if not isinstance(subsections, list):
            return False, "'subsection_analysis' must be a list"
        
        for i, subsection in enumerate(subsections):
            subsection_fields = ['document', 'refined_text', 'page_number']
            for field in subsection_fields:
                if field not in subsection:
                    return False, f"subsection_analysis[{i}] missing '{field}'"
        
        return True, "Valid format"
    
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"

def test_challenge_1b():
    """Test Challenge 1B solution"""
    print("Testing Challenge 1B Solution...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Copy sample PDFs from Challenge 1B collections
        collections = ["Collection 1", "Collection 2", "Collection 3"]
        test_collection = None
        
        for collection in collections:
            collection_path = Path("../Challenge_1b") / collection / "PDFs"
            if collection_path.exists():
                test_collection = collection
                pdf_files = list(collection_path.glob("*.pdf"))[:3]  # Limit to 3 PDFs for testing
                for pdf_file in pdf_files:
                    shutil.copy2(pdf_file, input_dir)
                break
        
        if not test_collection:
            print("Warning: No sample PDFs found. Please add PDFs to test.")
            return False
        
        print(f"Using test collection: {test_collection}")
        
        # Build Docker image
        print("Building Docker image...")
        build_result = subprocess.run([
            "docker", "build", "--platform", "linux/amd64", 
            "-t", "challenge1b:test", "."
        ], capture_output=True, text=True)
        
        if build_result.returncode != 0:
            print(f"Docker build failed: {build_result.stderr}")
            return False
        
        # Run Docker container with environment variables
        print("Running Docker container...")
        run_result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{input_dir}:/app/input:ro",
            "-v", f"{output_dir}:/app/output",
            "-e", "PERSONA=Travel Planner",
            "-e", "JOB_TO_BE_DONE=Plan a comprehensive trip for college friends",
            "--network", "none",
            "challenge1b:test"
        ], capture_output=True, text=True, timeout=120)
        
        if run_result.returncode != 0:
            print(f"Docker run failed: {run_result.stderr}")
            print(f"Docker run stdout: {run_result.stdout}")
            return False
        
        # Validate output
        print("Validating output...")
        output_file = output_dir / "challenge1b_output.json"
        
        if not output_file.exists():
            print("Missing output file: challenge1b_output.json")
            return False
        
        is_valid, message = validate_output_format(output_file)
        if is_valid:
            print(f"✓ challenge1b_output.json: {message}")
            
            # Show summary of results
            with open(output_file, 'r') as f:
                data = json.load(f)
                print(f"  - Processed {len(data['metadata']['input_documents'])} documents")
                print(f"  - Found {len(data['extracted_sections'])} relevant sections")
                print(f"  - Generated {len(data['subsection_analysis'])} refined analyses")
            
            return True
        else:
            print(f"✗ challenge1b_output.json: {message}")
            return False

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    success = test_challenge_1b()
    sys.exit(0 if success else 1)
