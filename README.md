# Challenge 1B: Persona-Driven Document Intelligence

## Overview
This solution builds upon Challenge 1A to create an intelligent document analyst that extracts and prioritizes relevant sections from PDF collections based on specific personas and their job-to-be-done.

## Architecture

### Integration with Challenge 1A
The solution integrates Challenge 1A by:
1. **Reusing the outline extraction**: Uses Challenge 1A's `process_pdfs.py` to extract structured document outlines
2. **Leveraging structure information**: Uses headings and titles from Challenge 1A to better understand document structure
3. **Building upon the foundation**: Challenge 1A provides the "understanding" of document structure that Challenge 1B uses for intelligent analysis

### Two-Stage Processing
1. **Stage 1 (Challenge 1A)**: Extract document outlines
   - Title extraction
   - Heading identification (H1, H2, H3)
   - Page-level structure mapping

2. **Stage 2 (Challenge 1B)**: Persona-driven analysis
   - Section relevance scoring using DialoGPT-small model
   - Content prioritization based on persona + job requirements
   - Refined text generation for selected sections

## Technical Implementation

### Dependencies
- **PyPDF2**: PDF text extraction (shared with Challenge 1A)
- **DialoGPT-small**: Lightweight conversational model (~351MB)
- **PyTorch**: CPU-only inference
- **Transformers**: Model loading and inference

### Processing Pipeline
1. **Document Collection Processing**: Processes 3-10 PDFs per collection
2. **Outline Integration**: Runs Challenge 1A internally to get structured outlines
3. **Section Extraction**: Uses outline information to extract meaningful sections
4. **Relevance Analysis**: AI-powered scoring based on persona and job requirements
5. **Content Refinement**: Generates tailored summaries for top sections

### Model Choice: DialoGPT-small
- **Size**: ~351MB (well under 1GB constraint)
- **Performance**: CPU-optimized for fast inference
- **Offline**: No internet dependency during execution
- **Task Suitability**: Good for text analysis and summarization tasks

## Usage

### Docker Execution
```bash
# Build
docker build --platform linux/amd64 -t challenge1b:latest .

# Run
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -e PERSONA="Travel Planner" \
  -e JOB_TO_BE_DONE="Plan a comprehensive trip for college friends" \
  --network none \
  challenge1b:latest
```

### Environment Variables
- `PERSONA`: The role/expertise of the target user (default: "Travel Planner")
- `JOB_TO_BE_DONE`: The specific task to accomplish (default: "Plan a comprehensive trip for a group of college friends.")

## Output Format
Generates `challenge1b_output.json` with:
- **Metadata**: Input documents, persona, job, timestamp
- **Extracted Sections**: Top 5 relevant sections with importance ranking
- **Subsection Analysis**: Top 3 sections with AI-generated refined text

## Performance Characteristics
- **Processing Time**: <60 seconds for 3-5 documents
- **Model Size**: ~351MB (within 1GB constraint)
- **CPU-Only**: No GPU dependencies
- **Offline**: No internet access required during execution

## Integration Benefits
By building on Challenge 1A:
1. **Structural Understanding**: Uses document outlines for better section identification
2. **Modular Design**: Clear separation between structure extraction and intelligence
3. **Reusability**: Challenge 1A can be used independently or as a component
4. **Efficiency**: Leverages existing PDF processing logic
5. **Reliability**: Fallback mechanisms if Challenge 1A is unavailable
