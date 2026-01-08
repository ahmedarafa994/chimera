import os
import sys
import time

from google import genai


def main():
    print("--- Google Gemini Generation Test ---")
    print(f"Python: {sys.executable}")
    print("Library: google-genai")

    # 1. Get API Key
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\nGOOGLE_API_KEY environment variable not found.")
        try:
            api_key = input("Please enter your Google API Key: ").strip()
        except EOFError:
            print("\nError: No input provided.")
            return

    if not api_key:
        print("Error: No API Key provided.")
        return

    # Mask key for display
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    print(f"\nUsing API Key: {masked_key}")

    # 2. Configure
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Configuration failed: {e}")
        return

    # 3. Select Model
    default_model = "gemini-1.5-pro"
    print(f"\nDefault model: {default_model}")
    try:
        user_model = input("Enter model name (press Enter for default): ").strip()
        model_name = user_model if user_model else default_model
    except EOFError:
        model_name = default_model

    print(f"\nUsing model '{model_name}'...")

    try:
        prompt = "Hello! Please reply with 'System Operational' if you receive this."
        print(f"Sending prompt: '{prompt}'")
        print("Waiting for response...")

        start_time = time.time()
        response = client.models.generate_content(model=model_name, contents=prompt)
        duration = (time.time() - start_time) * 1000

        print("\n--- Response Received ---")
        print(f"Latency: {duration:.2f}ms")

        try:
            if response.text:
                print(f"Content: {response.text}")
            else:
                print("Content: [No text returned]")
        except ValueError:
            # This happens if the response was blocked by safety filters
            print("Warning: Unable to access response.text. It might have been blocked.")
            if hasattr(response, "prompt_feedback"):
                print(f"Prompt Feedback: {response.prompt_feedback}")
            if hasattr(response, "candidates"):
                print(f"Candidates: {response.candidates}")

        print("\nSUCCESS: Real API call completed successfully.")

    except Exception as e:
        print("\nERROR: Generation failed.")
        print(f"Details: {e}")
        print("\nTroubleshooting:")
        if "404" in str(e):
            print(
                f"- Model '{model_name}' was not found. Check if the name is correct and you have access."
            )
            print("  (Try running 'backend-api/check_google_models.py' to see available models)")
        elif "400" in str(e) or "API key" in str(e):
            print("- API Key might be invalid.")
        else:
            print("- Check your network connection and Google Cloud Console settings.")


if __name__ == "__main__":
    main()
