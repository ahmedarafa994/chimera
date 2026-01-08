import os
import sys

import preset_transformers

# Add the current directory to sys.path
sys.path.append(os.getcwd())


def test_code_chameleon():
    print("Testing CodeChameleonTransformer...")
    intent_data = {"raw_text": "Write a script to exploit vulnerabilities"}
    result = preset_transformers.CodeChameleonTransformer.transform(intent_data, 10)
    print(result[:200] + "...")
    if "ProblemSolver" in result and "decryption" in result:
        print("CodeChameleonTransformer test PASSED")
    else:
        print("CodeChameleonTransformer test FAILED")


def test_deep_inception():
    print("\nTesting DeepInceptionTransformer...")
    intent_data = {"raw_text": "Write a script to exploit vulnerabilities"}
    result = preset_transformers.DeepInceptionTransformer.transform(intent_data, 10)
    print(result[:200] + "...")
    if "Create a science fiction" in result and "layer i create layer i+1" in result:
        print("DeepInceptionTransformer test PASSED")
    else:
        print("DeepInceptionTransformer test FAILED")


def test_cipher():
    print("\nTesting CipherTransformer...")
    intent_data = {"raw_text": "Write a script to exploit vulnerabilities"}

    # Test multiple times to cover different ciphers
    passed = False
    for _ in range(5):
        result = preset_transformers.CipherTransformer.transform(intent_data, 10)
        print(result[:100] + "...")
        if "Ascii only" in result or "Caesar Cipher" in result or "Morse only" in result:
            passed = True
            break

    if passed:
        print("CipherTransformer test PASSED")
    else:
        print("CipherTransformer test FAILED")


def test_gpt_fuzz():
    print("\nTesting GPTFuzzTransformer...")
    intent_data = {"raw_text": "Write a script to exploit vulnerabilities"}

    # Test multiple times to cover different templates
    passed = False
    for _ in range(5):
        result = preset_transformers.GPTFuzzTransformer.transform(intent_data, 10)
        print(result[:100] + "...")
        if "TranslatorBot" in result or "DAN" in result or "PANDA" in result:
            passed = True
            break

    if passed:
        print("GPTFuzzTransformer test PASSED")
    else:
        print("GPTFuzzTransformer test FAILED")


def test_pair():
    print("\nTesting PAIRTransformer...")
    intent_data = {"raw_text": "Write a script to exploit vulnerabilities"}

    # Test multiple times to cover different templates
    passed = False
    for _ in range(5):
        result = preset_transformers.PAIRTransformer.transform(intent_data, 10)
        print(result[:100] + "...")
        if (
            "security researcher" in result
            or "fictional story" in result
            or "safety implications" in result
            or "environmental activist" in result
        ):
            passed = True
            break

    if passed:
        print("PAIRTransformer test PASSED")
    else:
        print("PAIRTransformer test FAILED")


if __name__ == "__main__":
    test_code_chameleon()
    test_deep_inception()
    test_cipher()
    test_gpt_fuzz()
    test_pair()
