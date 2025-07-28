import os
import json
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class PersonaDrivenDocumentAnalyzer:
    def __init__(self):
        """Initialize the analyzer with lightweight model"""
        self.model_id = "microsoft/DialoGPT-small"  # ~351MB, fits within 1GB constraint
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the lightweight model and tokenizer"""
        try:
            print("Loading DialoGPT-small model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=True  # Force offline operation
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                local_files_only=True  # Force offline operation
            )
            self.model = self.model.to('cpu')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def run_challenge_1a(self, input_dir: str, output_dir: str) -> Dict[str, Dict]:
        """Run Challenge 1a processing to get document outlines"""
        print("Running Challenge 1a to extract document outlines...")
        
        # Path to Challenge 1a script
        challenge_1a_script = Path(__file__).parent / "challenge_1a" / "process_pdfs.py"
        
        if not challenge_1a_script.exists():
            print("Warning: Challenge 1a script not found, using fallback extraction")
            return self._fallback_outline_extraction(input_dir)
        
        # Run Challenge 1a directly on input/output directories
        try:
            result = subprocess.run([
                sys.executable, str(challenge_1a_script), input_dir, output_dir
            ], 
            cwd=Path(__file__).parent,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent / "challenge_1a")},
            capture_output=True, 
            text=True,
            timeout=60
            )
            
            if result.returncode != 0:
                print(f"Challenge 1a failed: {result.stderr}")
                print(f"Challenge 1a stdout: {result.stdout}")
                return self._fallback_outline_extraction(input_dir)
        
        except subprocess.TimeoutExpired:
            print("Challenge 1a timed out, using fallback")
            return self._fallback_outline_extraction(input_dir)
        except Exception as e:
            print(f"Error running Challenge 1a: {e}")
            return self._fallback_outline_extraction(input_dir)
        
        # Load the results from JSON files
        outlines = {}
        output_path = Path(output_dir)
        json_files = list(output_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files from Challenge 1a")
        
        for json_file in json_files:
            if json_file.name == "challenge1b_output.json":
                continue  # Skip our own output file
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    outline_data = json.load(f)
                    pdf_name = f"{json_file.stem}.pdf"
                    outlines[pdf_name] = outline_data
                    print(f"Loaded outline for {pdf_name}: {len(outline_data.get('outline', []))} sections")
            except Exception as e:
                print(f"Error loading outline for {json_file}: {e}")
        
        print(f"Total outlines loaded: {len(outlines)}")
        return outlines
    
    def _fallback_outline_extraction(self, input_dir: str) -> Dict[str, Dict]:
        """Fallback method if Challenge 1a is not available"""
        from PyPDF2 import PdfReader
        
        outlines = {}
        for pdf_file in Path(input_dir).glob("*.pdf"):
            try:
                reader = PdfReader(str(pdf_file))
                # Simple extraction
                first_page = reader.pages[0].extract_text() if reader.pages else ""
                lines = [line.strip() for line in first_page.split('\n') if line.strip()]
                
                title = lines[0] if lines else "Untitled Document"
                outline = [
                    {"level": "H1", "text": "Introduction", "page": 1},
                    {"level": "H2", "text": "Main Content", "page": 1}
                ]
                
                outlines[pdf_file.name] = {
                    "title": title,
                    "outline": outline
                }
            except Exception as e:
                print(f"Error in fallback extraction for {pdf_file}: {e}")
        
        return outlines
    
    def extract_sections_from_outlines(self, outlines: Dict[str, Dict], input_dir: str) -> List[Dict[str, Any]]:
        """Extract sections using outline information and full page content"""
        sections = []
        
        for pdf_name, outline_data in outlines.items():
            pdf_path = Path(input_dir) / pdf_name
            if not pdf_path.exists():
                continue
            
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(pdf_path))
                
                # Extract full text by page
                page_texts = {}
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():  # Only include pages with content
                        page_texts[page_num] = text
                
                # Create sections based on outline headings with full page content
                for heading in outline_data.get('outline', []):
                    page_num = heading.get('page', 1)
                    heading_text = heading.get('text', '')
                    level = heading.get('level', 'H1')
                    
                    # Get full content from the page
                    page_content = page_texts.get(page_num, '')
                    
                    # Split content into meaningful chunks (paragraphs)
                    paragraphs = [p.strip() for p in page_content.split('\n\n') if p.strip()]
                    
                    # Create section with substantial content
                    sections.append({
                        'document': pdf_name,
                        'title': heading_text,
                        'level': level,
                        'page_number': page_num,
                        'content': page_content,
                        'paragraphs': paragraphs,
                        'doc_title': outline_data.get('title', 'Untitled')
                    })
                
                # Also add sections for pages without headings but with content
                for page_num, content in page_texts.items():
                    if content and not any(s['page_number'] == page_num for s in sections if s['document'] == pdf_name):
                        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                        if paragraphs:  # Only add if there are substantial paragraphs
                            sections.append({
                                'document': pdf_name,
                                'title': f"Content from Page {page_num}",
                                'level': 'H2',
                                'page_number': page_num,
                                'content': content,
                                'paragraphs': paragraphs,
                                'doc_title': outline_data.get('title', 'Untitled')
                            })
                    
            except Exception as e:
                print(f"Error extracting sections from {pdf_name}: {e}")
        
        return sections
    
    def _extract_fallback_content(self, input_dir: str, persona: str, job: str) -> List[Dict[str, Any]]:
        """Fallback content extraction when outline-based extraction fails"""
        sections = []
        
        for pdf_file in Path(input_dir).glob("*.pdf"):
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(pdf_file))
                
                # Extract all content by page
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        # Split into meaningful chunks
                        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
                        
                        # Create sections from substantial paragraphs
                        for i, paragraph in enumerate(paragraphs[:3]):  # Max 3 per page
                            sections.append({
                                'document': pdf_file.name,
                                'title': f"Content from {pdf_file.stem} - Page {page_num} - Section {i+1}",
                                'level': 'H2',
                                'page_number': page_num,
                                'content': paragraph,
                                'paragraphs': [paragraph],
                                'doc_title': pdf_file.stem.replace('_', ' ').replace('-', ' ')
                            })
                
            except Exception as e:
                print(f"Error in fallback content extraction for {pdf_file}: {e}")
        
        # Analyze with model to get relevance scores
        if sections:
            sections = self.analyze_relevance_with_model(persona, job, sections, max_sections=8)
        
        return sections
    
    def analyze_relevance_with_model(self, persona: str, job: str, sections: List[Dict], max_sections: int = 10) -> List[Dict]:
        """Use the model to analyze section relevance and generate refined content"""
        relevant_sections = []
        
        # Simple keyword-based relevance scoring as primary method
        persona_keywords = persona.lower().split()
        job_keywords = job.lower().split()
        all_keywords = persona_keywords + job_keywords
        
        print(f"Analyzing {len(sections)} sections for relevance...")
        
        for i, section in enumerate(sections):
            try:
                # Calculate keyword-based relevance score
                title_text = section.get('title', '').lower()
                content_text = section.get('content', '').lower()
                combined_text = f"{title_text} {content_text}"
                
                # Count keyword matches
                keyword_matches = sum(1 for keyword in all_keywords if keyword in combined_text)
                base_score = min(10, 3 + keyword_matches)  # Base score 3-10
                
                # Try AI model for refinement (but don't rely on it) - with timeout
                final_score = base_score
                try:
                    if i < 20:  # Only use AI for first 20 sections to save time
                        relevance_prompt = f"""Rate relevance for {persona}: {job}
Title: {section['title']}
Content: {section['content'][:150]}...
Score 1-10:"""
                        
                        inputs = self.tokenizer(relevance_prompt, return_tensors="pt", max_length=250, truncation=True)
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=5,
                                do_sample=False,  # More deterministic
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id
                            )
                        
                        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                        
                        # Extract score from response
                        score_match = re.search(r'(\d+)', response)
                        ai_score = int(score_match.group(1)) if score_match and int(score_match.group(1)) <= 10 else base_score
                        
                        # Use average of keyword and AI score
                        final_score = (base_score + ai_score) // 2
                        
                except Exception as e:
                    print(f"AI scoring failed for section {i+1}, using keyword score: {e}")
                    final_score = base_score  # Fall back to keyword score
                
                # Include sections with score >= 4 (lower threshold)
                if final_score >= 4:
                    section['relevance_score'] = final_score
                    section['keyword_matches'] = keyword_matches
                    relevant_sections.append(section)
                
            except Exception as e:
                print(f"Error analyzing section relevance for section {i+1}: {e}")
                # Include section with default score if analysis fails
                section['relevance_score'] = 5
                relevant_sections.append(section)
        
        # Sort by relevance score and limit
        relevant_sections.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Ensure we have at least some content
        if len(relevant_sections) < 3:
            # Add more sections with lower scores
            for section in sections:
                if section not in relevant_sections:
                    section['relevance_score'] = 3
                    relevant_sections.append(section)
                    if len(relevant_sections) >= 5:
                        break
        
        return relevant_sections[:max_sections]
    
    def generate_refined_text(self, persona: str, job: str, section_content: str) -> str:
        """Generate refined text for subsection analysis using AI model"""
        
        # Try AI generation first
        try:
            # Create a focused prompt for content refinement
            prompt = f"""Summarize key points for {persona}:
{section_content[:600]}

Key points:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=400, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            refined_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            # Clean up the response
            refined_text = refined_text.strip()
            
            # Check if we got useful output
            if len(refined_text) > 20 and not refined_text.startswith(prompt[:20]):
                return refined_text
                
        except Exception as e:
            print(f"Error generating refined text: {e}")
        
        # Fallback: Use rule-based extraction
        # Extract the most informative sentences
        sentences = [s.strip() for s in section_content.split('.') if len(s.strip()) > 30]
        
        # Select sentences containing key terms
        persona_terms = persona.lower().split()
        job_terms = job.lower().split()
        key_terms = persona_terms + job_terms + ['guide', 'tips', 'how', 'best', 'important', 'should', 'can', 'will']
        
        relevant_sentences = []
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in key_terms):
                relevant_sentences.append(sentence.strip() + '.')
                if len(relevant_sentences) >= 3:
                    break
        
        if relevant_sentences:
            return ' '.join(relevant_sentences)
        else:
            # Final fallback: return first few sentences
            return '. '.join(sentences[:2]) + '.' if sentences else section_content[:300]
    
    def process_document_collection(self, input_dir: str, output_file: str, persona: str, job: str):
        """Main processing function that uses Challenge 1a output"""
        start_time = time.time()
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found in input directory")
            return
        
        print(f"Processing {len(pdf_files)} PDF files...")
        
        # Step 1: Run Challenge 1a to get document outlines
        output_dir = Path(output_file).parent
        outlines = self.run_challenge_1a(input_dir, str(output_dir))
        
        # Step 2: Extract sections using outline information
        all_sections = self.extract_sections_from_outlines(outlines, input_dir)
        
        print(f"Found {len(all_sections)} sections across all documents")
        
        # Step 3: Analyze relevance using the model (only if we have sections)
        if all_sections:
            relevant_sections = self.analyze_relevance_with_model(persona, job, all_sections)
        else:
            print("No sections found, using fallback content extraction")
            relevant_sections = self._extract_fallback_content(input_dir, persona, job)
        
        print(f"Selected {len(relevant_sections)} relevant sections for analysis")
        
        # Step 4: Generate output in required format
        output_data = {
            "metadata": {
                "input_documents": pdf_files,
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Format extracted sections
        for i, section in enumerate(relevant_sections[:5], 1):  # Top 5 sections
            output_data["extracted_sections"].append({
                "document": section['document'],
                "section_title": section['title'],
                "importance_rank": i,
                "page_number": section['page_number']
            })
        
        # Generate subsection analysis
        for section in relevant_sections[:3]:  # Top 3 for detailed analysis
            refined_text = self.generate_refined_text(persona, job, section['content'])
            output_data["subsection_analysis"].append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Output saved to {output_file}")

def main():
    """Main execution function"""
    # Default paths for Docker environment
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # For local testing
    if not os.path.exists(input_dir):
        input_dir = "./input"
        output_dir = "./output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read persona and job from environment or use defaults
    persona = os.getenv('PERSONA', 'Travel Planner')
    job = os.getenv('JOB_TO_BE_DONE', 'Plan a comprehensive trip for a group of college friends.')
    
    # Initialize analyzer
    analyzer = PersonaDrivenDocumentAnalyzer()
    
    # Process documents
    output_file = os.path.join(output_dir, "challenge1b_output.json")
    analyzer.process_document_collection(input_dir, output_file, persona, job)

if __name__ == "__main__":
    main()
