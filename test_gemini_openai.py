#!/usr/bin/env python3
"""
Test script for Gemini API using OpenAI's Python client
This demonstrates the code you provided with proper error handling
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


def test_basic_completion():
    """Test basic chat completion"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Chat Completion")
    print("=" * 60)

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "user", "content": "What is the capital of France? Answer in one word."}
            ],
        )
        print(f"\nResponse: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")

    except Exception as e:
        print(f"Error: {e}")


def test_with_reasoning():
    """Test chat completion with reasoning (your original code)"""
    print("\n" + "=" * 60)
    print("TEST 2: Chat Completion with High Reasoning")
    print("=" * 60)

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            reasoning_effort="high",  # Experimental reasoning mode
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain to me how AI works"},
            ],
        )

        print("\nResponse:")
        print(response.choices[0].message.content)
        print(f"\nModel: {response.model}")
        print(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")

    except Exception as e:
        print(f"Error: {e}")


def test_math_reasoning():
    """Test reasoning with a math problem"""
    print("\n" + "=" * 60)
    print("TEST 3: Math Problem with Reasoning")
    print("=" * 60)

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            reasoning_effort="high",
            messages=[
                {
                    "role": "system",
                    "content": "You are a math tutor. Show your step-by-step reasoning.",
                },
                {
                    "role": "user",
                    "content": "If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed for the entire journey?",
                },
            ],
        )

        print("\nResponse:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"Error: {e}")


def test_streaming():
    """Test streaming responses"""
    print("\n" + "=" * 60)
    print("TEST 4: Streaming Response")
    print("=" * 60)

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    try:
        print("\nStreaming response: ", end="", flush=True)
        stream = client.chat.completions.create(
            model="gemini-1.5-flash",  # Using flash for faster streaming
            messages=[{"role": "user", "content": "Write a three-line poem about coding"}],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"\nError: {e}")


def test_comparison():
    """Compare responses with and without reasoning"""
    print("\n" + "=" * 60)
    print("TEST 5: Comparison - With vs Without Reasoning")
    print("=" * 60)

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    question = "What are the top 3 benefits of using Python for AI development?"

    try:
        # Without reasoning
        print("\nWithout reasoning_effort:")
        response1 = client.chat.completions.create(
            model="gemini-2.5-pro", messages=[{"role": "user", "content": question}], max_tokens=200
        )
        print(response1.choices[0].message.content)

        # With high reasoning
        print("\n" + "-" * 60)
        print("With reasoning_effort='high':")
        response2 = client.chat.completions.create(
            model="gemini-2.5-pro",
            reasoning_effort="high",
            messages=[{"role": "user", "content": question}],
            max_tokens=200,
        )
        print(response2.choices[0].message.content)

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all tests"""
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("\n" + "!" * 60)
        print("ERROR: GEMINI_API_KEY not set!")
        print("!" * 60)
        print("\nPlease set your Gemini API key:")
        print("1. Get your key from: https://makersuite.google.com/app/apikey")
        print("2. Add to .env file: GEMINI_API_KEY=your-actual-key")
        print("3. Or set environment variable:")
        print("   Windows: set GEMINI_API_KEY=your-actual-key")
        print("   Linux/Mac: export GEMINI_API_KEY=your-actual-key")
        return

    print("\n" + "=" * 60)
    print("GEMINI API TESTING WITH OPENAI CLIENT")
    print("=" * 60)
    print(f"API Key found: {api_key[:20]}...")

    # Run tests
    try:
        test_basic_completion()
        test_with_reasoning()
        test_math_reasoning()
        test_streaming()
        test_comparison()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
