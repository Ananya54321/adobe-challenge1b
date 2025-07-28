import os
import json
import re
from pathlib import Path
from PyPDF2 import PdfReader
import time
from typing import List, Dict, Any

class PDFOutlineExtractor:
    def __init__(self):
        """Initialize the PDF outline extractor"""
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text content from PDF with page information"""
        try:
            reader = PdfReader(pdf_path)
            pages_content = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages_content.append({
                        'page_number': page_num,
                        'content': text.strip()
                    })
            
            return {
                'filename': os.path.basename(pdf_path),
                'pages': pages_content,
                'total_pages': len(pages_content)
            }
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return {'filename': os.path.basename(pdf_path), 'pages': [], 'total_pages': 0}
    
    def detect_title(self, pages_content: List[Dict]) -> str:
        """Extract document title from first few pages"""
        if not pages_content:
            return "Untitled Document"
        
        # Look at first page content
        first_page = pages_content[0]['content']
        lines = [line.strip() for line in first_page.split('\n') if line.strip()]
        
        if not lines:
            return "Untitled Document"
        
        # Try to find title - usually the first meaningful line
        for line in lines:
            # Skip very short lines, page numbers, headers/footers
            if len(line) > 5 and not re.match(r'^\d+$', line) and not line.lower().startswith('page'):
                # Clean up the line
                title = re.sub(r'\s+', ' ', line)
                if len(title) < 100:  # Reasonable title length
                    return title
        
        # Fallback to first line
        return lines[0] if lines else "Untitled Document"
    
    def identify_headings(self, pages_content: List[Dict]) -> List[Dict[str, Any]]:
        """Identify headings with their levels"""
        headings = []
        
        for page_data in pages_content:
            page_num = page_data['page_number']
            content = page_data['content']
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                heading_info = self.classify_heading(line, lines, i)
                if heading_info:
                    headings.append({
                        'level': heading_info['level'],
                        'text': line,
                        'page': page_num
                    })
        
        return headings
    
    def classify_heading(self, line: str, all_lines: List[str], line_index: int) -> Dict[str, str]:
        """Classify if a line is a heading and determine its level"""
        # Skip very long lines (likely paragraphs)
        if len(line) > 150:
            return None
        
        # Skip lines that look like body text
        if line.endswith('.') and len(line) > 50:
            return None
        
        # Numbered headings (1., 1.1, 1.1.1, etc.)
        numbered_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)', line)
        if numbered_match:
            level_parts = numbered_match.group(1).split('.')
            if len(level_parts) == 1:
                return {'level': 'H1'}
            elif len(level_parts) == 2:
                return {'level': 'H2'}
            else:
                return {'level': 'H3'}
        
        # All caps (likely H1)
        if line.isupper() and len(line) > 3 and len(line) < 80:
            return {'level': 'H1'}
        
        # Title case with no ending punctuation
        if (line.istitle() or re.match(r'^[A-Z][^.!?]*[^.!?]$', line)) and len(line) < 80:
            # Try to determine level based on context
            if line_index == 0 or (line_index > 0 and not all_lines[line_index-1].strip()):
                # Check if it's followed by more content (indicates heading)
                if line_index < len(all_lines) - 1 and all_lines[line_index + 1].strip():
                    # Simple heuristic: shorter = higher level
                    if len(line) < 30:
                        return {'level': 'H1'}
                    elif len(line) < 50:
                        return {'level': 'H2'}
                    else:
                        return {'level': 'H3'}
        
        # Lines that start with capital and are relatively short
        if (re.match(r'^[A-Z][a-zA-Z\s]+$', line) and 
            len(line) < 60 and 
            not line.endswith('.') and
            len(line.split()) <= 8):
            
            # Determine level based on length and position
            if len(line) < 25:
                return {'level': 'H1'}
            elif len(line) < 40:
                return {'level': 'H2'}
            else:
                return {'level': 'H3'}
        
        return None
    
    def process_pdf(self, pdf_path: str, output_path: str):
        """Process a single PDF and generate outline JSON"""
        start_time = time.time()
        
        # Extract text from PDF
        doc_data = self.extract_text_from_pdf(pdf_path)
        
        if not doc_data['pages']:
            print(f"No content extracted from {pdf_path}")
            return
        
        # Extract title
        title = self.detect_title(doc_data['pages'])
        
        # Identify headings
        headings = self.identify_headings(doc_data['pages'])
        
        # Create output structure
        output_data = {
            "title": title,
            "outline": headings
        }
        
        # Save output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        print(f"Processed {os.path.basename(pdf_path)} in {processing_time:.2f}s -> {os.path.basename(output_path)}")

def main():
    """Main execution function"""
    # Docker paths
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # For local testing
    if not input_dir.exists():
        input_dir = Path("./input")
        output_dir = Path("./output")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = PDFOutlineExtractor()
    
    # Process all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in input directory")
        return
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    for pdf_file in pdf_files:
        output_file = output_dir / f"{pdf_file.stem}.json"
        extractor.process_pdf(str(pdf_file), str(output_file))
    
    print("Processing completed!")

if __name__ == "__main__":
    main()