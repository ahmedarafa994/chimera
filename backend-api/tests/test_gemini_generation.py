import os
import time

from google import genai


def main() -> None:
    # 1. Get API Key
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = input("Please enter your Google API Key: ").strip()
        except EOFError:
            return

    if not api_key:
        return

    # Mask key for display
    f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"

    # 2. Configure
    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        return

    # 3. Select Model
    default_model = "gemini-1.5-pro"
    try:
        user_model = input("Enter model name (press Enter for default): ").strip()
        model_name = user_model if user_model else default_model
    except EOFError:
        model_name = default_model

    try:
        prompt = "Hello! Please reply with 'System Operational' if you receive this."

        start_time = time.time()
        response = client.models.generate_content(model=model_name, contents=prompt)
        (time.time() - start_time) * 1000

        try:
            if response.text:
                pass
            else:
                pass
        except ValueError:
            # This happens if the response was blocked by safety filters
            if hasattr(response, "prompt_feedback"):
                pass
            if hasattr(response, "candidates"):
                pass

    except Exception as e:
        if "404" in str(e) or "400" in str(e) or "API key" in str(e):
            pass
        else:
            pass


if __name__ == "__main__":
    main()
