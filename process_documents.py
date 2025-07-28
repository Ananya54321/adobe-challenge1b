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
    
    def _extract_keywords_from_context(self, persona: str, job: str) -> Dict[str, List[str]]:
        """Extract relevant keywords dynamically based on persona and job context"""
        # Convert inputs to lowercase for processing
        persona_lower = persona.lower()
        job_lower = job.lower()
        combined_context = f"{persona_lower} {job_lower}"
        
        # Base actionable terms that apply to most scenarios
        base_action_terms = [
            'how to', 'steps', 'process', 'guide', 'tutorial', 'tips', 'best practices',
            'examples', 'methods', 'techniques', 'strategies', 'approach', 'solution'
        ]
        
        # Base practical terms
        base_practical_terms = [
            'practical', 'useful', 'important', 'essential', 'key', 'main', 'primary',
            'recommended', 'effective', 'efficient', 'optimal', 'best', 'top'
        ]
        
        # Dynamic high-value terms based on context
        high_value_terms = base_action_terms.copy()
        medium_value_terms = base_practical_terms.copy()
        context_specific_terms = []
        
        # Analyze persona for domain-specific terms
        if any(term in persona_lower for term in ['travel', 'planner', 'tourist', 'vacation']):
            high_value_terms.extend(['activities', 'attractions', 'restaurants', 'hotels', 'things to do'])
            medium_value_terms.extend(['visit', 'explore', 'experience', 'discover'])
            context_specific_terms.extend(['nightlife', 'entertainment', 'sightseeing', 'tours'])
            
        elif any(term in persona_lower for term in ['developer', 'programmer', 'engineer', 'tech']):
            high_value_terms.extend(['code', 'implementation', 'setup', 'configuration', 'api'])
            medium_value_terms.extend(['build', 'create', 'develop', 'install', 'deploy'])
            context_specific_terms.extend(['debugging', 'testing', 'optimization', 'framework'])
            
        elif any(term in persona_lower for term in ['student', 'learner', 'educator', 'teacher']):
            high_value_terms.extend(['learn', 'study', 'understand', 'explain', 'concept'])
            medium_value_terms.extend(['knowledge', 'information', 'details', 'facts'])
            context_specific_terms.extend(['examples', 'exercises', 'practice', 'theory'])
            
        elif any(term in persona_lower for term in ['business', 'manager', 'analyst', 'professional']):
            high_value_terms.extend(['strategy', 'plan', 'process', 'workflow', 'management'])
            medium_value_terms.extend(['analyze', 'evaluate', 'assess', 'improve'])
            context_specific_terms.extend(['productivity', 'efficiency', 'optimization', 'metrics'])
            
        elif any(term in persona_lower for term in ['designer', 'creative', 'artist']):
            high_value_terms.extend(['design', 'create', 'visual', 'layout', 'style'])
            medium_value_terms.extend(['aesthetic', 'appearance', 'color', 'typography'])
            context_specific_terms.extend(['inspiration', 'creativity', 'composition', 'elements'])
        
        # Analyze job for task-specific terms
        job_keywords = job_lower.split()
        for keyword in job_keywords:
            if len(keyword) > 3:  # Only meaningful keywords
                context_specific_terms.append(keyword)
        
        # Add terms based on common job patterns
        if any(term in job_lower for term in ['plan', 'planning', 'organize']):
            high_value_terms.extend(['schedule', 'timeline', 'preparation', 'checklist'])
            
        elif any(term in job_lower for term in ['learn', 'understand', 'master']):
            high_value_terms.extend(['tutorial', 'guide', 'instruction', 'explanation'])
            
        elif any(term in job_lower for term in ['create', 'build', 'develop']):
            high_value_terms.extend(['implementation', 'construction', 'assembly', 'setup'])
            
        elif any(term in job_lower for term in ['analyze', 'research', 'investigate']):
            high_value_terms.extend(['data', 'information', 'findings', 'results'])
        
        return {
            'high_value': list(set(high_value_terms)),
            'medium_value': list(set(medium_value_terms)),
            'context_specific': list(set(context_specific_terms)),
            'persona_keywords': persona_lower.split(),
            'job_keywords': job_lower.split()
        }
    
    def analyze_relevance_with_model(self, persona: str, job: str, sections: List[Dict], max_sections: int = 10) -> List[Dict]:
        """Use dynamic rule-based analysis to find most relevant content for any domain"""
        relevant_sections = []
        
        print(f"Analyzing {len(sections)} sections for relevance...")
        
        # Extract keywords dynamically based on persona and job
        keyword_groups = self._extract_keywords_from_context(persona, job)
        
        for section in sections:
            title_text = section.get('title', '').lower()
            content_text = section.get('content', '').lower()
            combined_text = f"{title_text} {content_text}"
            
            score = 0
            
            # HIGH VALUE: Context-relevant action terms (weight: 15)
            for term in keyword_groups['high_value']:
                if term in combined_text:
                    score += 15
            
            # MEDIUM VALUE: Practical and useful content (weight: 10)
            for term in keyword_groups['medium_value']:
                if term in combined_text:
                    score += 10
            
            # CONTEXT SPECIFIC: Domain/job-specific terms (weight: 12)
            for term in keyword_groups['context_specific']:
                if term in combined_text:
                    score += 12
            
            # PERSONA ALIGNMENT: Direct persona keyword matches (weight: 8)
            for keyword in keyword_groups['persona_keywords']:
                if len(keyword) > 2 and keyword in combined_text:
                    score += 8
            
            # JOB ALIGNMENT: Direct job keyword matches (weight: 10)
            for keyword in keyword_groups['job_keywords']:
                if len(keyword) > 2 and keyword in combined_text:
                    score += 10
            
            # BONUS: Structured content (lists, examples, steps) (weight: 5)
            structure_indicators = ['•', '-', '1.', '2.', '3.', 'step', 'example', 'note:']
            for indicator in structure_indicators:
                if indicator in section.get('content', ''):
                    score += 5
                    break  # Only count once per section
            
            # BONUS: Specific details and actionable content (weight: 3)
            detail_indicators = ['specific', 'detailed', 'exactly', 'precisely', 'procedure', 'instruction']
            for indicator in detail_indicators:
                if indicator in combined_text:
                    score += 3
            
            # PENALTY: Generic introductory content (unless contextually relevant)
            penalty_terms = ['introduction', 'overview', 'abstract', 'summary', 'general information']
            context_relevance = any(term in combined_text for term in keyword_groups['high_value'][:5])
            
            for term in penalty_terms:
                if term in title_text and not context_relevance:
                    score -= 8  # Penalty for non-relevant generic content
            
            # PENALTY: Overly academic or theoretical content (unless persona is academic)
            academic_terms = ['theory', 'theoretical', 'academic', 'research methodology', 'literature review']
            is_academic_persona = any(term in persona.lower() for term in ['student', 'researcher', 'academic', 'scholar'])
            
            if not is_academic_persona:
                for term in academic_terms:
                    if term in combined_text:
                        score -= 5
            
            # BOOST: Perfect title matches for job requirements
            job_words = [word for word in keyword_groups['job_keywords'] if len(word) > 3]
            title_job_matches = sum(1 for word in job_words if word in title_text)
            if title_job_matches >= 2:  # Multiple job keywords in title
                score += 20
            elif title_job_matches == 1:
                score += 10
            
            # Document diversity consideration
            doc_name = section.get('document', '').lower()
            # Boost score for documents that seem relevant to the domain
            for term in keyword_groups['context_specific'][:10]:  # Top 10 context terms
                if term in doc_name:
                    score += 3
                    break
            
            # Final score calculation
            final_score = max(0, score)  # No negative scores
            
            # Dynamic threshold based on overall score distribution
            if final_score >= 10:  # Lower threshold to be more inclusive
                section['relevance_score'] = final_score
                section['score_breakdown'] = score
                relevant_sections.append(section)
                print(f"  Selected: {section['title'][:50]}... (score: {final_score})")
        
        # Sort by relevance score
        relevant_sections.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # If we don't have enough high-scoring sections, lower the threshold
        if len(relevant_sections) < 3:
            print(f"Only {len(relevant_sections)} sections found with score >= 10, lowering threshold...")
            for section in sections:
                if section not in relevant_sections:
                    # Recalculate with lower standards
                    title_text = section.get('title', '').lower()
                    content_text = section.get('content', '').lower()
                    combined_text = f"{title_text} {content_text}"
                    
                    basic_score = 0
                    # Just look for any persona/job keyword matches
                    for keyword in keyword_groups['persona_keywords'] + keyword_groups['job_keywords']:
                        if len(keyword) > 2 and keyword in combined_text:
                            basic_score += 5
                    
                    if basic_score >= 5:
                        section['relevance_score'] = basic_score
                        relevant_sections.append(section)
        
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
        """Generate refined text for subsection analysis using AI model - generic approach"""
        
        # Extract keywords for this specific context
        keyword_groups = self._extract_keywords_from_context(persona, job)
        
        # Try AI generation first
        try:
            # Create a focused prompt that adapts to any domain
            prompt = f"""As a {persona} working on: {job}

Extract the most relevant and useful information from this content that would help accomplish the task:
{section_content[:500]}

Focus on actionable details, practical information, specific examples, or key insights that are directly applicable.

Refined content:"""
            
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
        
        # Enhanced fallback: Extract contextually relevant information
        sentences = [s.strip() for s in section_content.split('.') if len(s.strip()) > 20]
        
        # Build dynamic indicators based on context
        relevance_indicators = (
            keyword_groups['high_value'] + 
            keyword_groups['medium_value'] + 
            keyword_groups['context_specific'] +
            ['example', 'step', 'method', 'approach', 'technique', 'process', 'procedure']
        )
        
        # Score sentences by contextual relevance
        scored_sentences = []
        for sentence in sentences[:15]:  # Check first 15 sentences
            score = 0
            sentence_lower = sentence.lower()
            
            # Count context-relevant terms
            for indicator in relevance_indicators:
                if indicator in sentence_lower:
                    score += 2
            
            # Bonus for structured content
            if any(marker in sentence for marker in ['•', '-', ':', ';', 'step', 'first', 'second', 'then']):
                score += 2
            
            # Bonus for specific details (numbers, names, etc.)
            import re
            if re.search(r'\d+', sentence):  # Contains numbers
                score += 1
            
            # Bonus for capitalized terms (proper nouns, specific names)
            capitalized_words = len([word for word in sentence.split() if word[0].isupper() and len(word) > 2])
            score += min(capitalized_words, 3)
            
            # Bonus for actionable language
            action_words = ['should', 'must', 'need to', 'important', 'key', 'essential', 'recommended']
            for action in action_words:
                if action in sentence_lower:
                    score += 1
            
            if score > 0:
                scored_sentences.append((score, sentence.strip() + '.'))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if scored_sentences:
            # Take top sentences that fit within length limit
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
        
        # Final fallback: Look for any content with context keywords
        lines = section_content.split('\n')
        useful_lines = []
        
        for line in lines:
            line = line.strip()
            if line and any(indicator in line.lower() for indicator in relevance_indicators[:15]):
                useful_lines.append(line)
                if len(useful_lines) >= 4:
                    break
        
        if useful_lines:
            return ' '.join(useful_lines)
        else:
            # Absolute fallback: Return the most substantial part
            paragraphs = [p.strip() for p in section_content.split('\n\n') if len(p.strip()) > 50]
            if paragraphs:
                # Take the first substantial paragraph
                return paragraphs[0][:400] + ("..." if len(paragraphs[0]) > 400 else "")
            else:
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
