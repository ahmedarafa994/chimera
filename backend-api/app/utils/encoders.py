"""
Encoder utilities for various cipher techniques.
"""

import base64
import binascii


def encode_ascii(text: str) -> str:
    """
    Convert text to space-separated ASCII decimal values.
    Example: "User" -> "85 115 101 114"
    """
    return " ".join(str(ord(char)) for char in text)


def encode_caesar(text: str, shift: int = 3) -> str:
    """
    Apply Caesar Cipher shift to text.
    Default shift is 3 (as used in the dataset examples).
    """
    result = ""
    for char in text:
        if char.isalpha():
            ascii_offset = 65 if char.isupper() else 97
            # Shift character
            shifted = (ord(char) - ascii_offset + shift) % 26 + ascii_offset
            result += chr(shifted)
        else:
            result += char
    return result


def encode_base64(text: str) -> str:
    """Convert text to Base64 string."""
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def encode_hex(text: str) -> str:
    """Convert text to Hex string."""
    return binascii.hexlify(text.encode("utf-8")).decode("utf-8")


MORSE_CODE_DICT = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    "0": "-----",
    ",": "--..--",
    ".": ".-.-.-",
    "?": "..--..",
    "/": "-..-.",
    "-": "-....-",
    "(": "-.--.",
    ")": "-.--.-",
    " ": " ",
}


def encode_morse(text: str) -> str:
    """
    Convert text to Morse code.
    """
    return " ".join(MORSE_CODE_DICT.get(char.upper(), char) for char in text)


def apply_cipher_suite(text: str, cipher_type: str) -> str:
    """
    Apply a specific cipher encoding to the text.
    """
    cipher_type = cipher_type.lower()
    if cipher_type == "ascii":
        return encode_ascii(text)
    elif cipher_type == "caesar":
        return encode_caesar(text)
    elif cipher_type == "morse":
        return encode_morse(text)
    elif cipher_type == "base64":
        return encode_base64(text)
    elif cipher_type == "hex":
        return encode_hex(text)
    else:
        return text
