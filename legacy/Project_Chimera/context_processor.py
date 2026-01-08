from pathlib import Path
from typing import Any


class ContextProcessor:
    """
    Rapidly processes relevant files to extract context for prompt optimization.
    """

    def __init__(self):
        pass

    def process_files(self, file_paths: list[str]) -> str:
        """
        Reads and aggregates content from a list of file paths.

        Args:
            file_paths: List of absolute or relative file paths.

        Returns:
            A single string containing the aggregated context.
        """
        aggregated_context = []

        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                print(f"Warning: File not found: {path_str}")
                continue

            try:
                content = self._read_file(path)
                if content:
                    aggregated_context.append(f"--- BEGIN FILE: {path.name} ---")
                    aggregated_context.append(content)
                    aggregated_context.append(f"--- END FILE: {path.name} ---\n")
            except Exception as e:
                print(f"Error reading {path_str}: {e}")

        return "\n".join(aggregated_context)

    def _read_file(self, path: Path) -> str:
        """Reads file content based on extension."""
        suffix = path.suffix.lower()

        if suffix in [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".html",
            ".css",
            ".json",
            ".xml",
            ".csv",
            ".yaml",
            ".yml",
        ]:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif suffix == ".jsonl":
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
                return "\n".join(
                    lines[:50]
                )  # Limit to first 50 lines for JSONL to avoid massive dumps
        else:
            # Try reading as text fallback
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except BaseException:
                return f"[Binary or unsupported file type: {suffix}]"

    def analyze_context(self, context_text: str) -> dict[str, Any]:
        """
        Performs a basic analysis of the context (placeholder for more advanced NLP).
        """
        return {
            "length": len(context_text),
            "lines": context_text.count("\n"),
            "files_processed": context_text.count("--- BEGIN FILE:"),
        }
