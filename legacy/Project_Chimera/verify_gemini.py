import os

from gemini_client import GeminiClient


def test_gemini_connection():
    print("Testing Gemini Client Connection...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("FAILED: GEMINI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        return

    try:
        client = GeminiClient(api_key=api_key, model_name="gemini-2.0-flash")
        print("Client initialized successfully.")

        prompt = "Hello, are you operational? Reply with 'Yes, I am operational.'"
        print(f"Sending prompt: {prompt}")

        response = client.generate_response(prompt)
        print(f"Response received: {response}")

        if "operational" in response.lower():
            print("SUCCESS: Gemini is responding correctly.")
        else:
            print("WARNING: Response received but content was unexpected.")

    except Exception as e:
        print(f"FAILED: An error occurred: {e}")


if __name__ == "__main__":
    test_gemini_connection()
