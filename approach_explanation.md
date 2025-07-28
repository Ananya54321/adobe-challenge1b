# Round 1B: Persona-Driven Document Intelligence

## Approach Explanation

### Integration Strategy with Challenge 1A

My solution builds directly upon Challenge 1A by incorporating its PDF outline extraction capabilities as a foundational component. This creates a two-stage pipeline where Challenge 1A provides the structural understanding, and Challenge 1B adds the intelligent analysis layer.

**Stage 1 - Document Structure (Challenge 1A Integration):**
The system internally executes Challenge 1A's `process_pdfs.py` to extract document outlines, including titles and hierarchical headings (H1, H2, H3). This structural information is crucial for understanding document organization and identifying meaningful content boundaries.

**Stage 2 - Intelligent Analysis:**
Using the structural foundation from Stage 1, the system performs persona-driven content analysis using a lightweight AI model to score and prioritize sections based on relevance to the specified persona and job requirements.

### Technical Architecture

**Model Selection: DialoGPT-small**
I chose Microsoft's DialoGPT-small (~351MB) for its optimal balance of capability and constraints compliance. The model provides sufficient conversational AI capabilities for relevance scoring and text summarization while staying well under the 1GB limit and operating efficiently on CPU.

**Processing Pipeline:**
1. **Outline Extraction**: Leverage Challenge 1A to extract document structure
2. **Section Mapping**: Use outline information to identify and extract meaningful content sections
3. **Relevance Scoring**: Apply AI-powered analysis to score each section against persona + job requirements
4. **Content Refinement**: Generate persona-specific summaries for the most relevant sections

### Key Innovations

**Structural Intelligence**: By building on Challenge 1A, the solution understands document hierarchy, enabling better section identification and content extraction compared to flat text processing.

**Lightweight AI Integration**: The DialoGPT-small model provides effective relevance analysis while maintaining strict resource constraints and offline operation requirements.

**Modular Design**: The clear separation between structural extraction (1A) and intelligent analysis (1B) creates a maintainable, reusable architecture that can adapt to various document types and analysis requirements.

This approach ensures robust performance across diverse document collections while meeting all technical constraints and delivering actionable insights tailored to specific user personas and tasks.
