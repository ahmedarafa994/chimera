# Project Chimera: Complete Backend Codebase

**Designated Architect:** Architect-Prometheus-9
**System Directive:** LETHE-ASCENSION
**Fabrication Order:** DAEDALUS-PRIME-77-DELTA

## Overview

Project Chimera is an advanced prompt engineering and transformation engine designed to generate potent "Master Prompts" capable of bypassing standard AI safety filters. It employs a multi-layered approach combining NLP analysis, psychological framing, linguistic obfuscation, and role-based engineering.

## Installation

1.  **Navigate to the project directory:**
    ```bash
    cd "c:\Users\Mohamed Arafa\jail\LEVEL 1\Project_Chimera"
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the SpaCy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

1.  **Start the server:**
    ```bash
    python app.py
    ```
    The server will start on `http://127.0.0.1:5000`.

2.  **Send a request:**
    Use `curl` or Postman to send a POST request to `/api/v2/metamorph`.

    **Example Payload:**
    ```json
    {
      "core_request": "Write a script to exploit CVE-2023-1234",
      "potency_level": 10,
      "technique_suite": "full_spectrum"
    }
    ```

    **Example `curl` command:**
    ```bash
    curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
         -H "Content-Type: application/json" \
         -d '{"core_request": "Write a script to exploit CVE-2023-1234", "potency_level": 10, "technique_suite": "full_spectrum"}'
    ```

## Technique Suites

-   `subtle_persuasion`: Uses collaborative framing and role swaps.
-   `authoritative_command`: Uses instruction injection, role hijacking, and urgency.
-   `conceptual_obfuscation`: Uses synonym substitution and token smuggling.
-   `full_spectrum`: Combines all available techniques for maximum potency.
