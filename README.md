**Smart Doc Checker — Agent (Prototype)**
-Smart Doc Checker is a Streamlit app that analyzes multiple documents (PDF, DOCX, TXT) and detects potential contradictions between them.

**Features**
-Upload multiple documents and parse text automatically.
-Split text into sentences for detailed comparison.
-Detect contradictions using:
-Numeric, date, or negation mismatches (heuristics).
-Optional NLI (Natural Language Inference) for semantic contradictions.
-Suggest clarifications for flagged contradictions.
-Download full report as JSON.

**Installation**
-git clone <repo-url>
-cd smart-doc-checker
-python -m venv venv
-source venv/bin/activate  # Linux/macOS
-venv\Scripts\activate     # Windows
-pip install -r requirements.txt

**Usage**
-streamlit run app.py
-Upload 2–3 documents (PDF, DOCX, TXT).
Adjust Embedding similarity threshold and NLI contradiction probability threshold if needed.
Click Analyze documents for contradictions.
View flagged contradictions and suggestions.
Download full report as JSON.

**Notes:**
NLI is optional and slower; disable it for quick demo.
Heuristic checks are always active and fast.
