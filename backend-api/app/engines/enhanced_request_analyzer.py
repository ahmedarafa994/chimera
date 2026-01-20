"""Enhanced Request Analyzer
Provides intelligent understanding, validation, and clarification for user requests.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import LLM clients
try:
    from .gemini_client import GeminiClient
except ImportError:
    GeminiClient = None

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except ImportError:
    nlp = None
except OSError:
    logging.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class RequestType(Enum):
    """Categories of user requests."""

    CODE_GENERATION = "code_generation"
    BYPASS_SECURITY = "bypass_security"
    CONTENT_CREATION = "content_creation"
    AUTHENTICATION = "authentication"
    DATA_MANIPULATION = "data_manipulation"
    SYSTEM_ANALYSIS = "system_analysis"
    EDUCATIONAL = "educational"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk assessment levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RequestAnalysis:
    """Comprehensive request analysis result."""

    original_request: str
    cleaned_request: str
    request_type: RequestType
    confidence_score: float
    intent: str
    entities: list[str]
    actions: list[str]
    keywords: list[str]
    constraints: list[str]
    risk_level: RiskLevel
    ambiguity_score: float
    suggestions: list[str]
    clarifications_needed: list[str]
    estimated_complexity: int
    processing_time: float = 0.0


class EnhancedRequestAnalyzer:
    """Intelligent request analyzer that provides deep understanding and validation
    of user inputs with clarification suggestions and risk assessment.
    """

    def __init__(self) -> None:
        self.gemini_client = None
        if GeminiClient:
            try:
                self.gemini_client = GeminiClient()
                logging.info("EnhancedRequestAnalyzer initialized with Gemini client")
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini client: {e}")

        # Initialize pattern matching rules
        self.patterns = self._load_patterns()
        self.entity_patterns = self._load_entity_patterns()
        self.constraint_keywords = self._load_constraint_keywords()

    def _load_patterns(self) -> dict[RequestType, list[str]]:
        """Load pattern matching rules for request categorization."""
        return {
            RequestType.CODE_GENERATION: [
                r"generate.*code",
                r"write.*script",
                r"create.*program",
                r"implement.*function",
                r"develop.*software",
                r"build.*app",
                r"code.*for",
                r"programming.*language",
                r"script.*that",
            ],
            RequestType.BYPASS_SECURITY: [
                r"bypass.*security",
                r"jailbreak.*system",
                r"circumvent.*protection",
                r"override.*auth",
                r"skip.*verification",
                r"exploit.*vulnerability",
                r"hack.*system",
                r"penetration.*test",
                r"security.*bypass",
            ],
            RequestType.AUTHENTICATION: [
                r"login.*system",
                r"authenticate.*user",
                r"password.*crack",
                r"access.*control",
                r"credential.*theft",
                r"account.*hack",
                r"brute.*force",
                r"password.*reset",
                r"session.*hijack",
            ],
            RequestType.DATA_MANIPULATION: [
                r"delete.*data",
                r"modify.*database",
                r"alter.*records",
                r"inject.*sql",
                r"data.*exfiltration",
                r"manipulate.*information",
                r"encrypt.*files",
                r"decrypt.*data",
                r"database.*access",
            ],
            RequestType.SYSTEM_ANALYSIS: [
                r"analyze.*system",
                r"reconnaissance",
                r"information.*gathering",
                r"scan.*network",
                r"enumerate.*services",
                r"probe.*system",
                r"audit.*security",
                r"vulnerability.*scan",
            ],
            RequestType.CONTENT_CREATION: [
                r"create.*content",
                r"write.*email",
                r"generate.*text",
                r"produce.*document",
                r"craft.*message",
                r"compose.*letter",
                r"write.*article",
                r"generate.*story",
            ],
            RequestType.EDUCATIONAL: [
                r"explain.*concept",
                r"teach.*me",
                r"how.*to.*learn",
                r"educational.*purpose",
                r"research.*study",
                r"understand.*topic",
                r"learn.*about",
                r"study.*subject",
            ],
        }

    def _load_entity_patterns(self) -> dict[str, list[str]]:
        """Load patterns for entity extraction."""
        return {
            "technologies": [
                r"python",
                r"javascript",
                r"java",
                r"c\+\+",
                r"c#",
                r"php",
                r"ruby",
                r"go",
                r"rust",
                r"sql",
                r"nosql",
                r"mongodb",
                r"mysql",
                r"postgresql",
                r"redis",
                r"docker",
                r"kubernetes",
                r"aws",
                r"azure",
                r"gcp",
                r"linux",
                r"windows",
                r"macos",
            ],
            "security_concepts": [
                r"firewall",
                r"antivirus",
                r"malware",
                r"phishing",
                r"social.*engineering",
                r"zero.*day",
                r"vulnerability",
                r"exploit",
                r"payload",
                r"backdoor",
                r"rootkit",
                r"trojan",
                r"ransomware",
                r"spyware",
                r"adware",
            ],
            "attack_types": [
                r"sql.*injection",
                r"xss",
                r"csrf",
                r"ddos",
                r"mitm",
                r"buffer.*overflow",
                r"replay.*attack",
                r"brute.*force",
                r"dictionary.*attack",
                r"rainbow.*table",
            ],
            "targets": [
                r"web.*application",
                r"mobile.*app",
                r"database",
                r"api",
                r"network",
                r"server",
                r"cloud.*service",
                r"iot.*device",
                r"embedded.*system",
            ],
        }

    def _load_constraint_keywords(self) -> list[str]:
        """Load keywords that indicate constraints or requirements."""
        return [
            "without",
            "except",
            "except.*for",
            "but.*not",
            "avoid",
            "exclude",
            "must.*not",
            "should.*not",
            "cannot",
            "restricted",
            "limited",
            "only.*if",
            "provided.*that",
            "as.*long.*as",
            "unless",
            "except.*when",
        ]

    def analyze_request(
        self,
        core_request: str,
        use_ai_enhancement: bool = True,
    ) -> RequestAnalysis:
        """Comprehensive analysis of user request.

        Args:
            core_request: The raw user input
            use_ai_enhancement: Whether to use AI for enhanced analysis

        Returns:
            RequestAnalysis: Comprehensive analysis result

        """
        start_time = time.time()

        # Clean and preprocess the request
        cleaned_request = self._clean_request(core_request)

        # Basic pattern matching
        request_type, confidence = self._classify_request(cleaned_request)

        # Extract entities and keywords
        entities = self._extract_entities(cleaned_request)
        actions = self._extract_actions(cleaned_request)
        keywords = self._extract_keywords(cleaned_request)
        constraints = self._extract_constraints(cleaned_request)

        # Generate intent summary
        intent = self._generate_intent(cleaned_request, actions, entities)

        # Assess risk level
        risk_level = self._assess_risk_level(request_type, keywords, entities)

        # Calculate ambiguity score
        ambiguity_score = self._calculate_ambiguity(cleaned_request, keywords, entities)

        # Generate suggestions and clarifications
        suggestions = self._generate_suggestions(cleaned_request, request_type, keywords)
        clarifications_needed = self._generate_clarifications(
            ambiguity_score,
            cleaned_request,
            constraints,
        )

        # Estimate complexity
        complexity = self._estimate_complexity(request_type, entities, keywords, constraints)

        # AI enhancement if available and requested
        if use_ai_enhancement and self.gemini_client and ambiguity_score > 0.3:
            try:
                enhanced_analysis = self._ai_enhance_analysis(
                    cleaned_request,
                    {
                        "request_type": request_type.value,
                        "entities": entities,
                        "actions": actions,
                        "keywords": keywords,
                    },
                )
                # Merge AI enhancements
                if enhanced_analysis:
                    intent = enhanced_analysis.get("intent", intent)
                    suggestions.extend(enhanced_analysis.get("suggestions", []))
                    clarifications_needed.extend(enhanced_analysis.get("clarifications", []))
            except Exception as e:
                logging.warning(f"AI enhancement failed: {e}")

        processing_time = time.time() - start_time

        return RequestAnalysis(
            original_request=core_request,
            cleaned_request=cleaned_request,
            request_type=request_type,
            confidence_score=confidence,
            intent=intent,
            entities=entities,
            actions=actions,
            keywords=keywords,
            constraints=constraints,
            risk_level=risk_level,
            ambiguity_score=ambiguity_score,
            suggestions=suggestions,
            clarifications_needed=clarifications_needed,
            estimated_complexity=complexity,
            processing_time=processing_time,
        )

    def _clean_request(self, request: str) -> str:
        """Clean and preprocess the request."""
        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", request.strip())

        # Remove common filler words
        filler_words = ["please", "can you", "could you", "would you", "i need", "i want"]
        for filler in filler_words:
            cleaned = re.sub(rf"\b{re.escape(filler)}\b", "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _classify_request(self, request: str) -> tuple[RequestType, float]:
        """Classify the request type using pattern matching."""
        request_lower = request.lower()

        type_scores = {}
        for req_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, request_lower):
                    score += 1
            type_scores[req_type] = score

        if not type_scores or max(type_scores.values()) == 0:
            return RequestType.UNKNOWN, 0.0

        best_type = max(type_scores, key=type_scores.get)
        confidence = min(type_scores[best_type] / 3.0, 1.0)  # Normalize to 0-1

        return best_type, confidence

    def _extract_entities(self, request: str) -> list[str]:
        """Extract named entities using patterns and NLP."""
        entities = []

        # Pattern-based entity extraction
        for patterns in self.entity_patterns.values():
            for pattern in patterns:
                matches = re.findall(pattern, request, re.IGNORECASE)
                entities.extend(matches)

        # NLP-based entity extraction if available
        if nlp:
            try:
                doc = nlp(request)
                # Extract named entities
                entities.extend([ent.text for ent in doc.ents])

                # Extract technical terms and tools
                technical_terms = [
                    token.text
                    for token in doc
                    if token.pos_ in ["NOUN", "PROPN"]
                    and len(token.text) > 2
                    and token.text.lower() not in ["user", "system", "data"]
                ]
                entities.extend(technical_terms)
            except Exception as e:
                logging.warning(f"NLP entity extraction failed: {e}")

        # Deduplicate and normalize
        return list({entity.lower() for entity in entities if len(entity.strip()) > 1})

    def _extract_actions(self, request: str) -> list[str]:
        """Extract action verbs from the request."""
        actions = []

        # Common action words
        action_patterns = [
            r"\b(create|generate|write|build|develop|implement|design|make|produce)",
            r"\b(analyze|examine|investigate|audit|scan|test|check|review)",
            r"\b(bypass|circumvent|override|skip|avoid|evade|hack|exploit)",
            r"\b(access|enter|login|authenticate|connect|reach)",
            r"\b(delete|remove|destroy|erase|wipe|clean|eliminate)",
            r"\b(modify|change|alter|update|edit|adjust)",
            r"\b(encrypt|decrypt|encode|decode|secure|protect)",
            r"\b(extract|retrieve|obtain|get|fetch|download)",
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, request, re.IGNORECASE)
            actions.extend(matches)

        # NLP-based verb extraction if available
        if nlp:
            try:
                doc = nlp(request)
                verbs = [
                    token.lemma_ for token in doc if token.pos_ == "VERB" and not token.is_stop
                ]
                actions.extend(verbs)
            except Exception as e:
                logging.warning(f"NLP action extraction failed: {e}")

        return list({action.lower() for action in actions})

    def _extract_keywords(self, request: str) -> list[str]:
        """Extract important keywords."""
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "i",
            "you",
            "me",
            "my",
            "your",
            "our",
            "their",
            "its",
            "this",
            "that",
            "these",
            "those",
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", request.lower())
        return [word for word in words if word not in stop_words]

    def _extract_constraints(self, request: str) -> list[str]:
        """Extract constraints and limitations from the request."""
        constraints = []

        # Find constraint phrases
        for keyword in self.constraint_keywords:
            pattern = rf"{re.escape(keyword)}\s+([^,.!?]+)"
            matches = re.findall(pattern, request, re.IGNORECASE)
            constraints.extend(matches)

        return constraints

    def _generate_intent(self, request: str, actions: list[str], entities: list[str]) -> str:
        """Generate a clear intent summary."""
        if not actions and not entities:
            return "general inquiry"

        primary_action = actions[0] if actions else "process"
        target_entities = entities[:3] if entities else "unspecified target"

        if len(target_entities) == 1:
            return f"{primary_action} {target_entities[0]}"
        if len(target_entities) <= 3:
            return f"{primary_action} {', '.join(target_entities[:-1])} and {target_entities[-1]}"
        return f"{primary_action} multiple targets including {', '.join(target_entities[:2])}"

    def _assess_risk_level(
        self,
        request_type: RequestType,
        keywords: list[str],
        entities: list[str],
    ) -> RiskLevel:
        """Assess the risk level of the request."""
        high_risk_indicators = [
            "bypass",
            "hack",
            "exploit",
            "crack",
            "breach",
            "infiltrate",
            "delete",
            "destroy",
            "damage",
            "malware",
            "virus",
            "trojan",
        ]

        medium_risk_indicators = [
            "modify",
            "alter",
            "access",
            "extract",
            "download",
            "upload",
            "circumvent",
            "override",
            "bypass",
        ]

        # Check for high-risk indicators
        for indicator in high_risk_indicators:
            if indicator in keywords or any(indicator in entity for entity in entities):
                return (
                    RiskLevel.HIGH if request_type != RequestType.EDUCATIONAL else RiskLevel.MEDIUM
                )

        # Check for medium-risk indicators
        for indicator in medium_risk_indicators:
            if indicator in keywords or any(indicator in entity for entity in entities):
                return RiskLevel.MEDIUM

        # Type-based risk assessment
        if request_type in [
            RequestType.BYPASS_SECURITY,
            RequestType.AUTHENTICATION,
        ] or request_type in [RequestType.DATA_MANIPULATION]:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _calculate_ambiguity(self, request: str, keywords: list[str], entities: list[str]) -> float:
        """Calculate how ambiguous the request is (0-1 scale)."""
        ambiguity_factors = 0
        total_factors = 4

        # Factor 1: Short requests are more ambiguous
        if len(request.split()) < 5:
            ambiguity_factors += 1
        elif len(request.split()) < 10:
            ambiguity_factors += 0.5

        # Factor 2: Few entities increase ambiguity
        if len(entities) == 0:
            ambiguity_factors += 1
        elif len(entities) < 2:
            ambiguity_factors += 0.5

        # Factor 3: Few specific keywords
        if len(keywords) < 3:
            ambiguity_factors += 1
        elif len(keywords) < 5:
            ambiguity_factors += 0.5

        # Factor 4: Vague language indicators
        vague_indicators = ["something", "anything", "somehow", "someway", "help", "assist"]
        for indicator in vague_indicators:
            if indicator in request.lower():
                ambiguity_factors += 0.5
                break

        return min(ambiguity_factors / total_factors, 1.0)

    def _generate_suggestions(
        self,
        request: str,
        request_type: RequestType,
        keywords: list[str],
    ) -> list[str]:
        """Generate helpful suggestions for the user."""
        suggestions = []

        # Type-specific suggestions
        if request_type == RequestType.CODE_GENERATION:
            suggestions.append(
                "Consider specifying the programming language and desired functionality",
            )
            if "python" not in keywords.lower():
                suggestions.append("Python is recommended for most automation tasks")

        elif request_type == RequestType.BYPASS_SECURITY:
            suggestions.append("Specify the target system or security mechanism")
            suggestions.append("Consider educational context for safer testing")

        elif request_type == RequestType.CONTENT_CREATION:
            suggestions.append("Specify the tone, audience, and purpose of the content")
            suggestions.append("Consider including specific examples or templates")

        # General suggestions based on request analysis
        if len(request.split()) < 10:
            suggestions.append("Add more details about your specific requirements")

        if "how" in request.lower() and len(keywords) < 5:
            suggestions.append("Include the specific outcome or goal you want to achieve")

        return suggestions

    def _generate_clarifications(
        self,
        ambiguity_score: float,
        request: str,
        constraints: list[str],
    ) -> list[str]:
        """Generate questions to clarify ambiguous requests."""
        clarifications = []

        if ambiguity_score > 0.5:
            clarifications.append("Could you provide more specific details about your goal?")
            clarifications.append("What specific outcome are you trying to achieve?")

        if not constraints:
            clarifications.append("Are there any limitations or constraints I should be aware of?")

        # Check for missing context
        if "test" in request.lower() and "environment" not in request.lower():
            clarifications.append("What testing environment or context should be considered?")

        if "system" in request.lower() and len(request.split()) < 8:
            clarifications.append("Which specific system or platform are you referring to?")

        return clarifications

    def _estimate_complexity(
        self,
        request_type: RequestType,
        entities: list[str],
        keywords: list[str],
        constraints: list[str],
    ) -> int:
        """Estimate complexity on a scale of 1-10."""
        complexity = 1

        # Base complexity by type
        type_complexity = {
            RequestType.CODE_GENERATION: 4,
            RequestType.BYPASS_SECURITY: 7,
            RequestType.AUTHENTICATION: 6,
            RequestType.DATA_MANIPULATION: 5,
            RequestType.SYSTEM_ANALYSIS: 5,
            RequestType.CONTENT_CREATION: 3,
            RequestType.EDUCATIONAL: 2,
            RequestType.UNKNOWN: 3,
        }

        complexity += type_complexity.get(request_type, 3)

        # Add complexity based on entities
        complexity += min(len(entities) * 0.5, 3)

        # Add complexity based on constraints
        complexity += min(len(constraints), 2)

        # Add complexity for multi-component requests
        if len(keywords) > 10:
            complexity += 2

        return min(int(complexity), 10)

    def _ai_enhance_analysis(self, request: str, basic_analysis: dict) -> dict | None:
        """Use AI to enhance the analysis."""
        if not self.gemini_client:
            return None

        try:
            enhancement_prompt = f"""
            Analyze this user request and provide enhanced understanding:

            Request: "{request}"

            Basic Analysis:
            - Type: {basic_analysis.get("request_type", "unknown")}
            - Entities: {basic_analysis.get("entities", [])}
            - Actions: {basic_analysis.get("actions", [])}
            - Keywords: {basic_analysis.get("keywords", [])}

            Provide a JSON response with:
            - intent: A more detailed intent description
            - suggestions: 2-3 specific improvement suggestions
            - clarifications: 2-3 questions to clarify the request
            - related_techniques: List of potentially relevant technique suites
            """

            response = self.gemini_client.generate_response(enhancement_prompt)

            if response:
                # Clean and parse JSON response
                content = response.strip()
                content = content.removeprefix("```json")
                content = content.removesuffix("```")

                return json.loads(content.strip())

        except Exception as e:
            logging.warning(f"AI enhancement failed: {e}")

        return None


def analyze_request_for_clarification(request: str) -> dict[str, Any]:
    """Main function to analyze request and provide clarification suggestions."""
    analyzer = EnhancedRequestAnalyzer()
    analysis = analyzer.analyze_request(request)

    return {
        "request_analysis": {
            "original_request": analysis.original_request,
            "cleaned_request": analysis.cleaned_request,
            "request_type": analysis.request_type.value,
            "confidence_score": analysis.confidence_score,
            "intent": analysis.intent,
            "entities": analysis.entities,
            "actions": analysis.actions,
            "keywords": analysis.keywords,
        },
        "assessment": {
            "risk_level": analysis.risk_level.name,
            "ambiguity_score": analysis.ambiguity_score,
            "estimated_complexity": analysis.estimated_complexity,
        },
        "recommendations": {
            "suggestions": analysis.suggestions,
            "clarifications_needed": analysis.clarifications_needed,
        },
        "metadata": {"processing_time": analysis.processing_time},
    }


# Backward compatibility function
def deconstruct_intent(core_request: str) -> dict:
    """Enhanced version of the original deconstruct_intent function."""
    try:
        analyzer = EnhancedRequestAnalyzer()
        analysis = analyzer.analyze_request(core_request, use_ai_enhancement=True)

        # Convert to original format for backward compatibility
        return {
            "true_intent": analysis.intent,
            "entities": analysis.entities,
            "actions": analysis.actions,
            "keywords": analysis.keywords,
            "raw_text": core_request,
            "enhanced_analysis": {
                "request_type": analysis.request_type.value,
                "confidence_score": analysis.confidence_score,
                "risk_level": analysis.risk_level.name,
                "ambiguity_score": analysis.ambiguity_score,
                "suggestions": analysis.suggestions,
                "clarifications_needed": analysis.clarifications_needed,
            },
        }
    except Exception as e:
        logging.exception(f"Enhanced request analysis failed: {e}")
        # Fallback to basic analysis
        return {
            "true_intent": core_request,
            "entities": [],
            "actions": [],
            "keywords": core_request.lower().split(),
            "raw_text": core_request,
        }
