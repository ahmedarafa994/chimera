"""Password Strength Validation Module.

Provides comprehensive password validation with:
- Minimum length requirements (12 characters)
- Character type requirements (uppercase, lowercase, digit, special)
- Entropy-based strength calculation
- Common password rejection
- Structured validation feedback

Security best practices based on NIST SP 800-63B guidelines.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Common Passwords List (Top passwords to reject)
# =============================================================================

# Based on common password lists - these are always rejected
COMMON_PASSWORDS: frozenset[str] = frozenset(
    [
        # Top commonly used passwords
        "password",
        "123456",
        "12345678",
        "1234567890",
        "qwerty",
        "abc123",
        "monkey",
        "1234567",
        "letmein",
        "trustno1",
        "dragon",
        "baseball",
        "iloveyou",
        "master",
        "sunshine",
        "ashley",
        "bailey",
        "passw0rd",
        "shadow",
        "123123",
        "654321",
        "superman",
        "qazwsx",
        "michael",
        "football",
        "password1",
        "password123",
        "welcome",
        "welcome1",
        "admin",
        "login",
        "starwars",
        "hello",
        "charlie",
        "donald",
        "password1234",
        "qwerty123",
        "qwertyuiop",
        "admin123",
        "root",
        "toor",
        "pass",
        "test",
        "guest",
        "master123",
        "changeme",
        "secret",
        "password!",
        "p@ssword",
        "p@ssw0rd",
        # Keyboard patterns
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm",
        "1qaz2wsx",
        "q1w2e3r4",
        "asdfasdf",
        "1q2w3e4r",
        "1234qwer",
        "qwer1234",
        "zaq1xsw2",
        # Simple patterns
        "aaaaaa",
        "111111",
        "000000",
        "abcdef",
        "123abc",
        "abc12345",
        "1234abcd",
        "password12",
        "passpass",
        "pass1234",
    ],
)


# =============================================================================
# Enums and Constants
# =============================================================================


class PasswordStrength(str, Enum):
    """Password strength levels based on entropy."""

    VERY_WEAK = "very_weak"  # Entropy < 28
    WEAK = "weak"  # Entropy 28-35
    FAIR = "fair"  # Entropy 36-59
    STRONG = "strong"  # Entropy 60-79
    VERY_STRONG = "very_strong"  # Entropy >= 80


class ValidationErrorType(str, Enum):
    """Types of password validation errors."""

    TOO_SHORT = "too_short"
    NO_UPPERCASE = "no_uppercase"
    NO_LOWERCASE = "no_lowercase"
    NO_DIGIT = "no_digit"
    NO_SPECIAL = "no_special"
    COMMON_PASSWORD = "common_password"
    LOW_ENTROPY = "low_entropy"
    CONTAINS_EMAIL = "contains_email"
    CONTAINS_USERNAME = "contains_username"
    SEQUENTIAL_CHARS = "sequential_chars"
    REPEATED_CHARS = "repeated_chars"


# Password requirements constants
MIN_PASSWORD_LENGTH = 12
MIN_ENTROPY_BITS = 50  # Minimum entropy for acceptance
SPECIAL_CHARACTERS = r"""!@#$%^&*()_+-=[]{}|;':",.<>?/`~"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ValidationError:
    """A single validation error."""

    error_type: ValidationErrorType
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class PasswordValidationResult:
    """Result of password validation.

    Attributes:
        is_valid: Whether the password passes all requirements
        strength: Password strength level
        entropy_bits: Calculated entropy in bits
        errors: List of validation errors
        warnings: List of validation warnings
        suggestions: List of suggestions to improve the password
        score: Numeric score from 0-100

    """

    is_valid: bool
    strength: PasswordStrength
    entropy_bits: float
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    score: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "is_valid": self.is_valid,
            "strength": self.strength.value,
            "entropy_bits": round(self.entropy_bits, 2),
            "score": self.score,
            "errors": [{"type": e.error_type.value, "message": e.message} for e in self.errors],
            "warnings": [{"type": w.error_type.value, "message": w.message} for w in self.warnings],
            "suggestions": self.suggestions,
        }


# =============================================================================
# Password Validator Service
# =============================================================================


class PasswordValidator:
    """Password validation service with entropy checking and strength analysis.

    Validates passwords against security best practices:
    - Minimum length of 12 characters (NIST recommendation)
    - Character complexity requirements
    - Common password checking
    - Entropy-based strength calculation
    - Pattern detection (sequential, repeated characters)

    Example:
        validator = PasswordValidator()
        result = validator.validate("MySecureP@ssw0rd123")
        if result.is_valid:
            print(f"Password strength: {result.strength.value}")
        else:
            for error in result.errors:
                print(f"Error: {error.message}")

    """

    def __init__(
        self,
        min_length: int = MIN_PASSWORD_LENGTH,
        min_entropy: float = MIN_ENTROPY_BITS,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digit: bool = True,
        require_special: bool = True,
        reject_common: bool = True,
    ) -> None:
        """Initialize password validator with configurable requirements.

        Args:
            min_length: Minimum password length (default: 12)
            min_entropy: Minimum entropy bits required (default: 50)
            require_uppercase: Require at least one uppercase letter
            require_lowercase: Require at least one lowercase letter
            require_digit: Require at least one digit
            require_special: Require at least one special character
            reject_common: Reject passwords in common password list

        """
        self.min_length = min_length
        self.min_entropy = min_entropy
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digit = require_digit
        self.require_special = require_special
        self.reject_common = reject_common

        # Precompile regex patterns for efficiency
        self._uppercase_pattern = re.compile(r"[A-Z]")
        self._lowercase_pattern = re.compile(r"[a-z]")
        self._digit_pattern = re.compile(r"\d")
        self._special_pattern = re.compile(r"[!@#$%^&*()_+\-=\[\]{}|;':\",./<>?`~\\]")
        self._repeated_pattern = re.compile(r"(.)\1{2,}")  # 3+ repeated chars
        self._sequential_pattern = re.compile(
            r"(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|"
            r"opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz|"
            r"012|123|234|345|456|567|678|789|890)",
            re.IGNORECASE,
        )

    def calculate_entropy(self, password: str) -> float:
        """Calculate password entropy in bits.

        Entropy = L * log2(N)
        Where:
            L = password length
            N = size of the character pool used

        Applies penalties for:
        - Repeated characters
        - Sequential patterns
        - Dictionary words

        Args:
            password: The password to analyze

        Returns:
            Entropy in bits (higher is better)

        """
        if not password:
            return 0.0

        # Determine character pool size
        pool_size = 0

        if self._lowercase_pattern.search(password):
            pool_size += 26  # lowercase letters
        if self._uppercase_pattern.search(password):
            pool_size += 26  # uppercase letters
        if self._digit_pattern.search(password):
            pool_size += 10  # digits
        if self._special_pattern.search(password):
            pool_size += 32  # special characters

        # Handle edge case of no recognized characters
        if pool_size == 0:
            # Use the unique character count as pool
            pool_size = len(set(password))
            if pool_size == 0:
                return 0.0

        # Base entropy calculation: L * log2(N)
        base_entropy = len(password) * math.log2(pool_size)

        # Apply penalties for patterns
        penalty = 0.0

        # Penalty for repeated characters (reduces effective length)
        repeated_matches = self._repeated_pattern.findall(password)
        if repeated_matches:
            penalty += len(repeated_matches) * 5

        # Penalty for sequential patterns
        sequential_matches = self._sequential_pattern.findall(password.lower())
        if sequential_matches:
            penalty += len(sequential_matches) * 8

        # Penalty if password is in common list (very significant)
        if password.lower() in COMMON_PASSWORDS:
            penalty += base_entropy * 0.7  # Reduce by 70%

        # Calculate unique character ratio (higher is better)
        unique_ratio = len(set(password.lower())) / len(password)
        if unique_ratio < 0.5:
            # Less than half unique characters - apply penalty
            penalty += (1 - unique_ratio) * 10

        return max(0.0, base_entropy - penalty)

    def get_strength(self, entropy_bits: float) -> PasswordStrength:
        """Determine password strength level from entropy.

        Thresholds based on NIST and security research:
        - Very Weak: < 28 bits (trivially crackable)
        - Weak: 28-35 bits (vulnerable to offline attacks)
        - Fair: 36-59 bits (resistant to most attacks)
        - Strong: 60-79 bits (resistant to advanced attacks)
        - Very Strong: >= 80 bits (cryptographically secure)

        Args:
            entropy_bits: Calculated entropy

        Returns:
            PasswordStrength enum value

        """
        if entropy_bits < 28:
            return PasswordStrength.VERY_WEAK
        if entropy_bits < 36:
            return PasswordStrength.WEAK
        if entropy_bits < 60:
            return PasswordStrength.FAIR
        if entropy_bits < 80:
            return PasswordStrength.STRONG
        return PasswordStrength.VERY_STRONG

    def calculate_score(self, entropy_bits: float, errors: list, warnings: list) -> int:
        """Calculate a 0-100 score for the password.

        Args:
            entropy_bits: Calculated entropy
            errors: List of validation errors
            warnings: List of validation warnings

        Returns:
            Score from 0 to 100

        """
        # Base score from entropy (0-70 points)
        # 80+ bits entropy = full 70 points
        entropy_score = min(70, (entropy_bits / 80) * 70)

        # Requirement compliance bonus (up to 30 points)
        requirement_score = 30

        # Deduct for errors (10 points each, max 30)
        error_deduction = min(30, len(errors) * 10)
        requirement_score -= error_deduction

        # Deduct for warnings (3 points each, max 15)
        warning_deduction = min(15, len(warnings) * 3)
        requirement_score = max(0, requirement_score - warning_deduction)

        return int(max(0, min(100, entropy_score + requirement_score)))

    def validate(
        self,
        password: str,
        email: str | None = None,
        username: str | None = None,
    ) -> PasswordValidationResult:
        """Validate a password against all requirements.

        Args:
            password: The password to validate
            email: Optional email to check password doesn't contain it
            username: Optional username to check password doesn't contain it

        Returns:
            PasswordValidationResult with validation status and feedback

        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        suggestions: list[str] = []

        # Ensure password is a string
        if not isinstance(password, str):
            password = str(password) if password else ""

        # Calculate entropy first
        entropy_bits = self.calculate_entropy(password)
        strength = self.get_strength(entropy_bits)

        # Length check
        if len(password) < self.min_length:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.TOO_SHORT,
                    message=f"Password must be at least {self.min_length} characters long",
                ),
            )
            suggestions.append(f"Add {self.min_length - len(password)} more characters")

        # Uppercase check
        if self.require_uppercase and not self._uppercase_pattern.search(password):
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.NO_UPPERCASE,
                    message="Password must contain at least one uppercase letter (A-Z)",
                ),
            )
            suggestions.append("Add an uppercase letter (A-Z)")

        # Lowercase check
        if self.require_lowercase and not self._lowercase_pattern.search(password):
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.NO_LOWERCASE,
                    message="Password must contain at least one lowercase letter (a-z)",
                ),
            )
            suggestions.append("Add a lowercase letter (a-z)")

        # Digit check
        if self.require_digit and not self._digit_pattern.search(password):
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.NO_DIGIT,
                    message="Password must contain at least one digit (0-9)",
                ),
            )
            suggestions.append("Add a digit (0-9)")

        # Special character check
        if self.require_special and not self._special_pattern.search(password):
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.NO_SPECIAL,
                    message=f"Password must contain at least one special character ({SPECIAL_CHARACTERS[:10]}...)",
                ),
            )
            suggestions.append("Add a special character (!@#$%^&* etc.)")

        # Common password check
        if self.reject_common and password.lower() in COMMON_PASSWORDS:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.COMMON_PASSWORD,
                    message="Password is too common and easily guessable",
                ),
            )
            suggestions.append("Choose a more unique password")

        # Entropy check
        if entropy_bits < self.min_entropy:
            warnings.append(
                ValidationError(
                    error_type=ValidationErrorType.LOW_ENTROPY,
                    message=f"Password entropy is low ({entropy_bits:.1f} bits). Consider making it more complex.",
                    severity="warning",
                ),
            )
            suggestions.append("Use a mix of random characters to increase strength")

        # Check if password contains email
        if email and len(email) > 3:
            email_parts = email.lower().split("@")
            for part in email_parts:
                if len(part) > 3 and part in password.lower():
                    warnings.append(
                        ValidationError(
                            error_type=ValidationErrorType.CONTAINS_EMAIL,
                            message="Password should not contain parts of your email address",
                            severity="warning",
                        ),
                    )
                    suggestions.append("Remove email-related content from password")
                    break

        # Check if password contains username
        if username and len(username) > 3 and username.lower() in password.lower():
            warnings.append(
                ValidationError(
                    error_type=ValidationErrorType.CONTAINS_USERNAME,
                    message="Password should not contain your username",
                    severity="warning",
                ),
            )
            suggestions.append("Remove your username from the password")

        # Check for sequential characters
        if self._sequential_pattern.search(password.lower()):
            warnings.append(
                ValidationError(
                    error_type=ValidationErrorType.SEQUENTIAL_CHARS,
                    message="Password contains sequential characters (abc, 123, etc.)",
                    severity="warning",
                ),
            )
            suggestions.append("Avoid sequential patterns like 'abc' or '123'")

        # Check for repeated characters
        if self._repeated_pattern.search(password):
            warnings.append(
                ValidationError(
                    error_type=ValidationErrorType.REPEATED_CHARS,
                    message="Password contains repeated characters (aaa, 111, etc.)",
                    severity="warning",
                ),
            )
            suggestions.append("Avoid repeating the same character multiple times")

        # Password is valid if there are no errors
        is_valid = len(errors) == 0

        # Calculate score
        score = self.calculate_score(entropy_bits, errors, warnings)

        # Adjust strength if invalid due to errors
        if not is_valid and strength.value not in (
            PasswordStrength.VERY_WEAK.value,
            PasswordStrength.WEAK.value,
        ):
            strength = PasswordStrength.WEAK

        # Remove duplicate suggestions
        suggestions = list(dict.fromkeys(suggestions))

        return PasswordValidationResult(
            is_valid=is_valid,
            strength=strength,
            entropy_bits=entropy_bits,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            score=score,
        )

    def is_valid(
        self,
        password: str,
        email: str | None = None,
        username: str | None = None,
    ) -> bool:
        """Simple validation check returning only True/False.

        Args:
            password: The password to validate
            email: Optional email to check
            username: Optional username to check

        Returns:
            True if password meets all requirements, False otherwise

        """
        return self.validate(password, email, username).is_valid


# =============================================================================
# Default Validator Instance
# =============================================================================

# Global validator instance with default settings
password_validator = PasswordValidator()


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_password(
    password: str,
    email: str | None = None,
    username: str | None = None,
) -> PasswordValidationResult:
    """Validate a password using the default validator.

    Args:
        password: The password to validate
        email: Optional email to check password doesn't contain it
        username: Optional username to check password doesn't contain it

    Returns:
        PasswordValidationResult with validation status and feedback

    """
    return password_validator.validate(password, email, username)


def is_password_valid(
    password: str,
    email: str | None = None,
    username: str | None = None,
) -> bool:
    """Check if a password is valid using the default validator.

    Args:
        password: The password to validate
        email: Optional email to check
        username: Optional username to check

    Returns:
        True if password meets all requirements, False otherwise

    """
    return password_validator.is_valid(password, email, username)


def get_password_strength(password: str) -> PasswordStrength:
    """Get the strength level of a password.

    Args:
        password: The password to analyze

    Returns:
        PasswordStrength enum value

    """
    entropy = password_validator.calculate_entropy(password)
    return password_validator.get_strength(entropy)


def calculate_password_entropy(password: str) -> float:
    """Calculate the entropy of a password in bits.

    Args:
        password: The password to analyze

    Returns:
        Entropy in bits

    """
    return password_validator.calculate_entropy(password)
