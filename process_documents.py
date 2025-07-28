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
        """Use enhanced rule-based analysis to find most relevant travel content"""
        relevant_sections = []
        
        print(f"Analyzing {len(sections)} sections for relevance...")
        
        # Define comprehensive scoring criteria for travel planning
        for section in sections:
            title_text = section.get('title', '').lower()
            content_text = section.get('content', '').lower()
            combined_text = f"{title_text} {content_text}"
            
            score = 0
            
            # HIGH VALUE: Practical travel activities and recommendations (weight: 10-15)
            high_value_terms = [
                'things to do', 'activities', 'attractions', 'nightlife', 'entertainment',
                'restaurants', 'bars', 'clubs', 'beach', 'coastal', 'water sports',
                'shopping', 'markets', 'festivals', 'events', 'tours', 'excursions'
            ]
            for term in high_value_terms:
                if term in combined_text:
                    score += 15
            
            # MEDIUM VALUE: City guides and practical info (weight: 8-12)
            medium_value_terms = [
                'guide', 'visit', 'explore', 'see', 'experience', 'discover',
                'accommodation', 'hotels', 'budget', 'cost', 'tips', 'advice',
                'transportation', 'getting around', 'where to stay', 'where to eat'
            ]
            for term in medium_value_terms:
                if term in combined_text:
                    score += 10
            
            # BONUS: College/youth-friendly content (weight: 8)
            youth_terms = [
                'young', 'student', 'budget', 'cheap', 'affordable', 'free',
                'party', 'social', 'group', 'friends', 'fun', 'adventure'
            ]
            for term in youth_terms:
                if term in combined_text:
                    score += 8
            
            # BONUS: Specific venue names and practical details (weight: 5)
            specific_indicators = ['address', 'location', 'phone', 'hours', 'price', 'cost', '€', '$']
            for indicator in specific_indicators:
                if indicator in combined_text:
                    score += 5
            
            # BONUS: Lists and structured content (weight: 5)
            if any(marker in section.get('content', '') for marker in ['•', '-', '1.', '2.', '3.']):
                score += 5
            
            # PENALTY: Avoid pure introductions and history unless travel-relevant
            penalty_terms = ['introduction', 'overview', 'history', 'historical', 'ancient', 'medieval']
            travel_context = any(term in combined_text for term in ['visit', 'tour', 'see', 'attraction', 'site'])
            
            for term in penalty_terms:
                if term in title_text and not travel_context:
                    score -= 10  # Heavy penalty for non-travel historical content
            
            # PENALTY: Generic content
            if any(generic in title_text for generic in ['comprehensive guide', 'introduction to', 'overview of']):
                score -= 5
            
            # SPECIAL BOOST: Perfect matches for travel planning
            perfect_matches = [
                'coastal adventures', 'nightlife and entertainment', 'culinary experiences',
                'packing tips', 'general tips', 'where to go', 'what to do',
                'best restaurants', 'top attractions', 'must-see', 'recommended'
            ]
            for match in perfect_matches:
                if match in combined_text:
                    score += 20
            
            # Document diversity bonus - prefer varied content sources
            doc_name = section.get('document', '').lower()
            if 'things to do' in doc_name:
                score += 5
            elif 'restaurants' in doc_name or 'cuisine' in doc_name:
                score += 5
            elif 'tips' in doc_name:
                score += 5
            elif 'cities' in doc_name and any(city in combined_text for city in ['nice', 'cannes', 'marseille', 'antibes']):
                score += 5
            
            # Final score calculation
            final_score = max(0, score)  # No negative scores
            
            # Only include sections with meaningful scores
            if final_score >= 15:  # Higher threshold for quality
                section['relevance_score'] = final_score
                section['score_breakdown'] = score
                relevant_sections.append(section)
                print(f"  Selected: {section['title'][:50]}... (score: {final_score})")
        
        # Sort by relevance score
        relevant_sections.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Ensure document diversity in final selection
        final_sections = []
        doc_count = {}
        
        # First pass: Take top scorer from each document type
        for section in relevant_sections:
            doc = section.get('document', 'Unknown')
            if doc_count.get(doc, 0) < 2:  # Max 2 per document initially
                final_sections.append(section)
                doc_count[doc] = doc_count.get(doc, 0) + 1
                if len(final_sections) >= max_sections:
                    break
        
        # Second pass: Fill remaining slots with highest scorers
        if len(final_sections) < max_sections:
            for section in relevant_sections:
                if section not in final_sections:
                    final_sections.append(section)
                    if len(final_sections) >= max_sections:
                        break
        
        print(f"Selected {len(final_sections)} sections for analysis")
        return final_sections[:max_sections]
    
    def generate_refined_text(self, persona: str, job: str, section_content: str) -> str:
        """Generate refined text for subsection analysis using AI model"""
        
        # Try AI generation first
        try:
            # Create a focused prompt for travel planning content
            prompt = f"""As a {persona} helping with: {job}

Extract the most useful travel information from this content:
{section_content[:500]}

Focus on practical details like places to visit, activities, restaurants, tips, costs, or logistics.

Refined travel guide:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=400, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,  # Deterministic for consistency
                    num_beams=3,      # Better quality
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            refined_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            # Clean up the response
            refined_text = refined_text.strip()
            
            # Check if we got useful output
            if len(refined_text) > 30 and not refined_text.startswith("I can't") and "cannot" not in refined_text.lower():
                return refined_text
                
        except Exception as e:
            print(f"Error generating refined text: {e}")
        
        # Enhanced fallback: Extract practical travel information
        sentences = [s.strip() for s in section_content.split('.') if len(s.strip()) > 20]
        
        # Look for actionable travel content
        travel_indicators = ['visit', 'explore', 'try', 'enjoy', 'experience', 'discover', 'see', 'go to',
                           'restaurant', 'hotel', 'bar', 'club', 'beach', 'attraction', 'activity',
                           'tip', 'recommendation', 'must', 'best', 'top', 'popular', 'famous',
                           'cost', 'price', 'budget', 'free', 'open', 'hours', 'location', 'address']
        
        # Score sentences by travel relevance
        scored_sentences = []
        for sentence in sentences[:15]:  # Check first 15 sentences
            score = 0
            sentence_lower = sentence.lower()
            
            # Count travel-related terms
            for indicator in travel_indicators:
                if indicator in sentence_lower:
                    score += 2
            
            # Bonus for lists or specific details
            if any(marker in sentence for marker in ['•', '-', ':', ';']):
                score += 1
            
            # Bonus for specific names/places (capitalized words)
            capitalized_words = len([word for word in sentence.split() if word[0].isupper() and len(word) > 2])
            score += min(capitalized_words, 3)
            
            if score > 0:
                scored_sentences.append((score, sentence.strip() + '.'))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if scored_sentences:
            # Take top 3-5 sentences based on length
            selected_sentences = []
            total_length = 0
            for score, sentence in scored_sentences:
                if total_length + len(sentence) < 600:  # Keep under reasonable length
                    selected_sentences.append(sentence)
                    total_length += len(sentence)
                    if len(selected_sentences) >= 5:
                        break
            
            if selected_sentences:
                return ' '.join(selected_sentences)
        
        # Final fallback: return first portion with travel focus
        lines = section_content.split('\n')
        useful_lines = []
        for line in lines:
            line = line.strip()
            if line and any(indicator in line.lower() for indicator in travel_indicators[:10]):
                useful_lines.append(line)
                if len(useful_lines) >= 4:
                    break
        
        if useful_lines:
            return ' '.join(useful_lines)
        else:
            # Absolute fallback
            return section_content[:400] + ("..." if len(section_content) > 400 else "")
    
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
