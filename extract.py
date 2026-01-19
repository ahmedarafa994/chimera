import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8")

pdf_path = "docs/prompt.pdf"

print(f"Attempting to extract from {pdf_path}")

try:
    print("Trying pypdf...")
    import pypdf

    print(f"pypdf version: {pypdf.__version__}")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        print(f"Extracting page {i}")
        text += page.extract_text() + "\n"
    print("Extraction successful with pypdf")
    print("-" * 20)
    print(text)
    sys.exit(0)
except ImportError:
    print("pypdf ImportError")
except Exception as e:
    print(f"pypdf failed: {e}")
    traceback.print_exc()

print("-" * 20)

try:
    print("Trying pdfplumber...")
    import pdfplumber

    print("pdfplumber imported")
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for i, page in enumerate(pdf.pages):
            print(f"Extracting page {i}")
            text += page.extract_text() + "\n"
        print("Extraction successful with pdfplumber")
        print(text)
    sys.exit(0)
except ImportError:
    print("pdfplumber ImportError")
except Exception as e:
    print(f"pdfplumber failed: {e}")
    traceback.print_exc()

sys.exit(1)
