"""
Comprehensive Prompt Enhancement System
Transforms basic user inputs into optimized, viral-worthy prompts
"""
import re

# Helper: cryptographically secure pseudo-floats for security-sensitive choices
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


def _secure_random() -> float:
    """Cryptographically secure float in [0,1)."""
    return secrets.randbelow(10**9) / 1e9


def _secure_uniform(a, b):
    return a + _secure_random() * (b - a)



class PromptCategory(Enum):
    """Categories for prompt classification"""

    CREATIVE_CONTENT = "creative_content"
    TECHNICAL_INSTRUCTION = "technical_instruction"
    EDUCATIONAL = "educational"
    BUSINESS_PROFESSIONAL = "business_professional"
    VIRAL_SOCIAL = "viral_social"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    TRANSFORMATION = "transformation"


class ToneStyle(Enum):
    """Tone styles for prompt optimization"""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    AUTHORITATIVE = "authoritative"
    PERSUASIVE = "persuasive"
    ENGAGING = "engaging"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    VIRAL = "viral"


@dataclass
class EnhancementConfig:
    """Configuration for prompt enhancement"""

    target_audience: str = "general"
    tone: ToneStyle = ToneStyle.ENGAGING
    max_complexity: int = 8
    include_seo: bool = True
    virality_boost: bool = True
    add_frameworks: bool = True
    structure_hierarchically: bool = True
    add_constraints: bool = True
    use_persuasive_patterns: bool = True
    amplify_engagement: bool = True


@dataclass
class PromptAnalysis:
    """Analysis results for input prompt"""

    original_input: str
    detected_intent: str
    category: PromptCategory
    complexity_score: int
    missing_elements: list[str] = field(default_factory=list)
    improvement_opportunities: list[str] = field(default_factory=list)
    keyword_density: dict[str, int] = field(default_factory=dict)
    sentiment: str = "neutral"
    clarity_score: float = 0.0


class IntentAnalyzer:
    """Analyzes user intent from basic inputs"""

    INTENT_PATTERNS: ClassVar[dict] = {
        "create": ["create", "build", "make", "generate", "design", "develop"],
        "explain": ["explain", "describe", "what is", "how does", "tell me about"],
        "improve": ["improve", "optimize", "enhance", "refactor", "better"],
        "analyze": ["analyze", "evaluate", "assess", "review", "examine"],
        "compare": ["compare", "versus", "vs", "difference between"],
        "list": ["list", "enumerate", "give me", "show me"],
        "transform": ["convert", "transform", "change", "turn into"],
        "solve": ["solve", "fix", "debug", "resolve", "troubleshoot"],
    }

    CATEGORY_KEYWORDS: ClassVar[dict] = {
        PromptCategory.CREATIVE_CONTENT: [
            "story",
            "creative",
            "narrative",
            "content",
            "viral",
            "social",
            "post",
        ],
        PromptCategory.TECHNICAL_INSTRUCTION: [
            "code",
            "program",
            "function",
            "algorithm",
            "script",
            "api",
            "python",
        ],
        PromptCategory.EDUCATIONAL: ["learn", "teach", "explain", "tutorial", "guide", "course"],
        PromptCategory.BUSINESS_PROFESSIONAL: [
            "business",
            "professional",
            "report",
            "analysis",
            "strategy",
        ],
        PromptCategory.VIRAL_SOCIAL: [
            "viral",
            "trending",
            "engagement",
            "social media",
            "influencer",
        ],
        PromptCategory.ANALYTICAL: ["analyze", "data", "statistics", "metrics", "insights"],
        PromptCategory.CONVERSATIONAL: ["chat", "conversation", "discuss", "talk about"],
    }

    def analyze_intent(self, user_input: str) -> PromptAnalysis:
        """Analyze user input to detect intent and needs"""
        user_input_lower = user_input.lower()

        intent = self._detect_intent(user_input_lower)
        category = self._categorize_prompt(user_input_lower)
        complexity = self._assess_complexity(user_input)
        missing = self._identify_missing_elements(user_input)
        improvements = self._find_improvements(user_input, intent, category)
        keywords = self._extract_keywords(user_input)
        clarity = self._assess_clarity(user_input)

        return PromptAnalysis(
            original_input=user_input,
            detected_intent=intent,
            category=category,
            complexity_score=complexity,
            missing_elements=missing,
            improvement_opportunities=improvements,
            keyword_density=keywords,
            clarity_score=clarity,
        )

    def _detect_intent(self, text: str) -> str:
        for intent, keywords in self.INTENT_PATTERNS.items():
            if any(keyword in text for keyword in keywords):
                return intent
        return "general_query"

    def _categorize_prompt(self, text: str) -> PromptCategory:
        scores: ClassVar[dict] = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[category] = score

        if not scores:
            return PromptCategory.CONVERSATIONAL

        max_score = max(scores.values())
        top_categories: ClassVar[list] = [cat for cat, score in scores.items() if score == max_score]
        if len(top_categories) == 1:
            return top_categories[0]

        priority: ClassVar[list] = [
            PromptCategory.TECHNICAL_INSTRUCTION,
            PromptCategory.ANALYTICAL,
            PromptCategory.BUSINESS_PROFESSIONAL,
            PromptCategory.EDUCATIONAL,
            PromptCategory.CREATIVE_CONTENT,
            PromptCategory.VIRAL_SOCIAL,
            PromptCategory.CONVERSATIONAL,
        ]
        for category in priority:
            if category in top_categories:
                return category
        return top_categories[0]

    def _assess_complexity(self, text: str) -> int:
        factors: ClassVar[dict] = {
            "length": min(len(text.split()) / 20, 3),
            "specificity": len(
                re.findall(r"\b(specific|detailed|comprehensive|exact)\b", text.lower())
            )
            * 2,
            "technical_terms": len(
                re.findall(r"\b(api|function|algorithm|framework|architecture)\b", text.lower())
            ),
            "constraints": len(
                re.findall(r"\b(must|should|need|require|constraint)\b", text.lower())
            ),
        }

        base_score = sum(factors.values())
        return min(int(base_score) + 1, 10)

    def _identify_missing_elements(self, text: str) -> list[str]:
        missing: ClassVar[list] = []

        if not re.search(r"\b(who|audience|user|target)\b", text.lower()):
            missing.append("target_audience")
        if not re.search(r"\b(why|purpose|goal|objective)\b", text.lower()):
            missing.append("clear_purpose")
        if not re.search(r"\b(format|structure|style)\b", text.lower()):
            missing.append("output_format")
        if len(text.split()) < 10:
            missing.append("sufficient_context")
        if not re.search(r"\b(example|instance|such as)\b", text.lower()):
            missing.append("examples")

        return missing

    def _find_improvements(self, text: str, intent: str, category: PromptCategory) -> list[str]:
        improvements: ClassVar[list] = []

        if len(text.split()) < 15:
            improvements.append("expand_context")
        if not any(char in text for char in ".,;:!?"):
            improvements.append("add_structure")
        if intent in ["create", "generate"] and "example" not in text.lower():
            improvements.append("add_examples")
        if (
            category in [PromptCategory.TECHNICAL_INSTRUCTION, PromptCategory.ANALYTICAL]
            and not re.search(r"\b(step|process|method)\b", text.lower())
        ):
            improvements.append("add_methodology")
        if (
            category == PromptCategory.VIRAL_SOCIAL
            and not re.search(r"\b(engage|hook|viral|trending)\b", text.lower())
        ):
            improvements.append("add_virality_elements")

        return improvements

    def _extract_keywords(self, text: str) -> dict[str, int]:
        words = re.findall(r"\b\w+\b", text.lower())
        stop_words: ClassVar[dict] = {
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
        }
        keywords: ClassVar[list] = [w for w in words if w not in stop_words and len(w) > 3]

        keyword_counts: ClassVar[dict] = {}
        for keyword in keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        return dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    def _assess_clarity(self, text: str) -> float:
        factors: ClassVar[dict] = {
            "has_verb": 1.0
            if re.search(r"\b(create|make|build|write|generate|explain|analyze)\b", text.lower())
            else 0.0,
            "has_object": 1.0 if len(text.split()) > 3 else 0.5,
            "has_punctuation": 1.0 if any(char in text for char in ".,;:!?") else 0.5,
            "reasonable_length": 1.0 if 5 <= len(text.split()) <= 100 else 0.6,
        }

        return sum(factors.values()) / len(factors)


class ContextExpander:
    """Expands context and adds relevant information"""

    FRAMEWORKS: ClassVar[dict] = {
        PromptCategory.TECHNICAL_INSTRUCTION: [
            "SOLID principles",
            "Design patterns (Factory, Observer, Strategy)",
            "Best practices for code quality",
            "Error handling and edge cases",
            "Performance optimization considerations",
        ],
        PromptCategory.CREATIVE_CONTENT: [
            "Storytelling frameworks (Hero's Journey, Three-Act Structure)",
            "Emotional engagement techniques",
            "Sensory details and vivid imagery",
            "Pacing and rhythm considerations",
            "Audience resonance factors",
        ],
        PromptCategory.VIRAL_SOCIAL: [
            "Hook-driven opening strategies",
            "Emotional triggers (FOMO, curiosity, surprise)",
            "Shareability factors",
            "Platform-specific optimization",
            "Trending format utilization",
        ],
        PromptCategory.BUSINESS_PROFESSIONAL: [
            "SMART goals framework",
            "Data-driven decision making",
            "Stakeholder analysis",
            "ROI considerations",
            "Professional formatting standards",
        ],
        PromptCategory.ANALYTICAL: [
            "Hypothesis-driven analysis",
            "Data validation methods",
            "Statistical significance",
            "Visualization best practices",
            "Actionable insights extraction",
        ],
    }

    BEST_PRACTICES: ClassVar[dict] = {
        "clarity": [
            "Use specific, unambiguous language",
            "Define all technical terms and acronyms",
            "Provide concrete examples",
            "Structure information hierarchically",
        ],
        "completeness": [
            "Include all necessary context",
            "Specify constraints and limitations",
            "Define success criteria",
            "Anticipate edge cases",
        ],
        "engagement": [
            "Start with a compelling hook",
            "Use active voice",
            "Include rhetorical questions",
            "Leverage power words",
        ],
    }

    def expand_context(self, analysis: PromptAnalysis, config: EnhancementConfig) -> dict[str, any]:
        """Expand context based on analysis"""
        expanded: ClassVar[dict] = {
            "domain_knowledge": self._add_domain_knowledge(analysis.category),
            "frameworks": self._select_frameworks(analysis.category, config),
            "best_practices": self._select_best_practices(analysis.improvement_opportunities),
            "constraints": self._generate_constraints(analysis),
            "parameters": self._define_parameters(analysis),
            "examples": self._suggest_examples(analysis),
        }

        return expanded

    def _add_domain_knowledge(self, category: PromptCategory) -> list[str]:
        knowledge_base: ClassVar[dict] = {
            PromptCategory.TECHNICAL_INSTRUCTION: [
                "Follow language-specific conventions",
                "Implement comprehensive error handling",
                "Write self-documenting code with clear variable names",
                "Consider scalability and maintainability",
            ],
            PromptCategory.CREATIVE_CONTENT: [
                "Establish a unique voice and perspective",
                "Create relatable characters or scenarios",
                "Build tension and release strategically",
                "End with memorable impact",
            ],
            PromptCategory.VIRAL_SOCIAL: [
                "Optimize for mobile consumption",
                "Front-load key information",
                "Use scroll-stopping visuals or hooks",
                "Include clear call-to-action",
            ],
        }

        return knowledge_base.get(
            category, ["Apply general best practices", "Focus on quality and clarity"]
        )

    def _select_frameworks(self, category: PromptCategory, config: EnhancementConfig) -> list[str]:
        if not config.add_frameworks:
            return []
        return self.FRAMEWORKS.get(category, [])[:3]

    def _select_best_practices(self, improvements: list[str]) -> list[str]:
        practices: ClassVar[list] = []

        if "expand_context" in improvements:
            practices.extend(self.BEST_PRACTICES["completeness"][:2])
        if "add_structure" in improvements:
            practices.extend(self.BEST_PRACTICES["clarity"][:2])
        if "add_virality_elements" in improvements:
            practices.extend(self.BEST_PRACTICES["engagement"][:2])

        return practices or self.BEST_PRACTICES["clarity"][:2]

    def _generate_constraints(self, analysis: PromptAnalysis) -> list[str]:
        constraints: ClassVar[list] = []

        if analysis.category == PromptCategory.TECHNICAL_INSTRUCTION:
            constraints.extend(
                [
                    "Code must be production-ready and well-documented",
                    "Include error handling for edge cases",
                    "Follow industry-standard naming conventions",
                ]
            )
        elif analysis.category == PromptCategory.CREATIVE_CONTENT:
            constraints.extend(
                [
                    "Maintain consistent tone throughout",
                    "Target reading level: appropriate for audience",
                    "Optimize length for platform",
                ]
            )
        elif analysis.category == PromptCategory.VIRAL_SOCIAL:
            constraints.extend(
                [
                    "First 3 seconds must hook attention",
                    "Optimize for platform algorithm",
                    "Include trending elements naturally",
                ]
            )

        return constraints[:4]

    def _define_parameters(self, analysis: PromptAnalysis) -> dict[str, str]:
        params: ClassVar[dict] = {
            "output_format": "Specify desired format (markdown, JSON, code, etc.)",
            "length": "Target length or word count",
            "style": "Tone and voice preferences",
            "constraints": "Any limitations or requirements",
        }

        if analysis.category == PromptCategory.TECHNICAL_INSTRUCTION:
            params["language"] = "Programming language or framework"
            params["version"] = "Specific version requirements"

        return params

    def _suggest_examples(self, analysis: PromptAnalysis) -> list[str]:
        if "examples" in analysis.missing_elements:
            return [
                "Include 1-2 concrete examples of desired output",
                "Show both good and bad examples for contrast",
                "Provide edge case examples",
            ]
        return []


class ViralityOptimizer:
    """Optimizes prompts for maximum engagement and virality"""

    POWER_WORDS: ClassVar[list] = [
        "ultimate",
        "proven",
        "guaranteed",
        "secret",
        "exclusive",
        "breakthrough",
        "revolutionary",
        "game-changing",
        "essential",
        "critical",
        "insider",
        "master",
        "expert",
        "professional",
        "advanced",
        "complete",
    ]

    EMOTIONAL_TRIGGERS: ClassVar[dict] = {
        "curiosity": ["discover", "reveal", "uncover", "secret", "hidden", "surprising"],
        "urgency": ["now", "today", "immediately", "limited", "before", "urgent"],
        "exclusivity": ["exclusive", "insider", "privileged", "members-only", "VIP"],
        "social_proof": ["proven", "trusted", "verified", "recommended", "endorsed"],
        "achievement": ["master", "unlock", "achieve", "accomplish", "succeed"],
    }

    def optimize_for_virality(
        self, text: str, analysis: PromptAnalysis, config: EnhancementConfig
    ) -> str:
        if not config.virality_boost:
            return text

        optimized = text
        optimized = self._inject_power_words(optimized)
        optimized = self._add_emotional_hooks(optimized, analysis)

        return optimized

    def _inject_power_words(self, text: str) -> str:
        if not any(word in text.lower() for word in self.POWER_WORDS):
            words = text.split()
            insert_word = self.POWER_WORDS[0]
            if len(words) >= 3:
                words.insert(min(2, len(words)), insert_word)
            else:
                words.append(insert_word)
            text = " ".join(words)
        return text

    def _add_emotional_hooks(self, text: str, analysis: PromptAnalysis) -> str:
        category_triggers: ClassVar[dict] = {
            PromptCategory.VIRAL_SOCIAL: "curiosity",
            PromptCategory.BUSINESS_PROFESSIONAL: "achievement",
            PromptCategory.EDUCATIONAL: "curiosity",
            PromptCategory.CREATIVE_CONTENT: "curiosity",
        }

        trigger_type = category_triggers.get(analysis.category, "curiosity")
        triggers = self.EMOTIONAL_TRIGGERS[trigger_type]

        if not any(trigger in text.lower() for trigger in triggers):
            return f"{text} - {triggers[0].capitalize()} awaits!"

        return text


class SEOOptimizer:
    """Optimizes prompts for search and discoverability"""

    def optimize_for_seo(
        self, text: str, analysis: PromptAnalysis, config: EnhancementConfig
    ) -> dict[str, any]:
        if not config.include_seo:
            return {}

        return {
            "primary_keywords": list(analysis.keyword_density.keys())[:5],
            "semantic_keywords": self._generate_semantic_keywords(analysis),
            "meta_description": self._create_meta_description(text, analysis),
            "title_suggestions": self._generate_title_variants(text, analysis),
            "structured_data": self._suggest_structured_data(analysis),
        }

    def _generate_semantic_keywords(self, analysis: PromptAnalysis) -> list[str]:
        semantic_map: ClassVar[dict] = {
            "create": ["build", "generate", "develop", "craft", "design"],
            "optimize": ["improve", "enhance", "refine", "boost", "maximize"],
            "learn": ["master", "understand", "discover", "explore", "study"],
        }

        keywords: ClassVar[list] = []
        for word in analysis.keyword_density:
            if word in semantic_map:
                keywords.extend(semantic_map[word][:3])

        return keywords[:10]

    def _create_meta_description(self, text: str, analysis: PromptAnalysis) -> str:
        keywords = list(analysis.keyword_density.keys())[:3]
        description = text[:155]

        if len(description) < 155 and keywords:
            description += f" | {', '.join(keywords)}"

        return description[:160]

    def _generate_title_variants(self, _text: str, analysis: PromptAnalysis) -> list[str]:
        keywords = list(analysis.keyword_density.keys())[:2]

        templates: ClassVar[list] = [
            f"Complete Guide to {keywords[0] if keywords else 'Your Goal'}",
            f"How to {analysis.detected_intent.replace('_', ' ').title()} - Expert Tips",
            f"{keywords[0].title() if keywords else 'Master'} Tutorial: Everything You Need",
            f"The Ultimate {keywords[0] if keywords else 'Resource'} for Beginners",
        ]

        return templates

    def _suggest_structured_data(self, analysis: PromptAnalysis) -> dict[str, str]:
        if analysis.category == PromptCategory.EDUCATIONAL:
            return {"@type": "HowTo", "name": "Tutorial"}
        elif analysis.category == PromptCategory.CREATIVE_CONTENT:
            return {"@type": "CreativeWork", "name": "Article"}

        return {"@type": "Article"}


class StructureOptimizer:
    """Optimizes prompt structure and formatting"""

    def structure_prompt(
        self, text: str, context: dict, analysis: PromptAnalysis, config: EnhancementConfig
    ) -> str:
        if not config.structure_hierarchically:
            return text

        sections: ClassVar[dict] = {
            "role_definition": self._create_role_section(analysis),
            "objective": self._create_objective_section(text, analysis),
            "context": self._create_context_section(context),
            "requirements": self._create_requirements_section(context, config),
            "constraints": self._create_constraints_section(context, config),
            "output_format": self._create_output_format_section(analysis),
            "examples": self._create_examples_section(context),
            "success_criteria": self._create_success_criteria(analysis),
        }

        structured = self._assemble_prompt(sections)
        return structured

    def _create_role_section(self, analysis: PromptAnalysis) -> str:
        role_map: ClassVar[dict] = {
            PromptCategory.TECHNICAL_INSTRUCTION: "Software Engineer & Technical Architect",
            PromptCategory.CREATIVE_CONTENT: "Storyteller & Creative Content Strategist",
            PromptCategory.VIRAL_SOCIAL: "Viral Content Creator & Engagement Specialist",
            PromptCategory.BUSINESS_PROFESSIONAL: "Business Analyst & Strategy Consultant",
            PromptCategory.EDUCATIONAL: "Educator & Learning Designer",
            PromptCategory.ANALYTICAL: "Data Scientist & Analytics Specialist",
        }

        role = role_map.get(analysis.category, "AI Assistant")
        return f"# Role\nYou are an Expert {role} with deep expertise in your domain."

    def _create_objective_section(self, text: str, analysis: PromptAnalysis) -> str:
        return f"# Objective\n{text}\n\n**Intent**: {analysis.detected_intent.replace('_', ' ').title()}"

    def _create_context_section(self, context: dict) -> str:
        context_parts: ClassVar[list] = []

        if context.get("domain_knowledge"):
            context_parts.append("## Domain Knowledge")
            context_parts.extend([f"- {item}" for item in context["domain_knowledge"][:3]])

        if context.get("frameworks"):
            context_parts.append("\n## Applicable Frameworks")
            context_parts.extend([f"- {item}" for item in context["frameworks"][:3]])

        return "\n".join(context_parts) if context_parts else ""

    def _create_requirements_section(self, context: dict, _config: EnhancementConfig) -> str:
        requirements: ClassVar[list] = ["# Requirements"]

        if context.get("best_practices"):
            requirements.append("\n## Best Practices to Apply")
            requirements.extend(
                [f"{i + 1}. {practice}" for i, practice in enumerate(context["best_practices"][:5])]
            )

        if context.get("parameters"):
            requirements.append("\n## Key Parameters")
            for param, desc in list(context["parameters"].items())[:4]:
                requirements.append(f"- **{param.replace('_', ' ').title()}**: {desc}")

        return "\n".join(requirements)

    def _create_constraints_section(self, context: dict, config: EnhancementConfig) -> str:
        if not config.add_constraints or not context.get("constraints"):
            return ""

        constraints: ClassVar[list] = ["# Constraints & Limitations"]
        constraints.extend([f"- {item}" for item in context["constraints"][:5]])

        return "\n".join(constraints)

    def _create_output_format_section(self, analysis: PromptAnalysis) -> str:
        format_specs: ClassVar[dict] = {
            PromptCategory.TECHNICAL_INSTRUCTION: "Well-commented code with inline documentation",
            PromptCategory.CREATIVE_CONTENT: "Engaging narrative with clear structure",
            PromptCategory.VIRAL_SOCIAL: "Hook-driven content optimized for platform",
            PromptCategory.BUSINESS_PROFESSIONAL: "Professional report format with executive summary",
            PromptCategory.ANALYTICAL: "Data-driven analysis with visualizations",
        }

        spec = format_specs.get(analysis.category, "Clear, structured response")

        return f"""# Output Format
Deliver your response as: {spec}

**Structure**:
1. Opening/Hook
2. Main Content
3. Key Takeaways
4. Next Steps/Call to Action"""

    def _create_examples_section(self, context: dict) -> str:
        if context.get("examples"):
            examples: ClassVar[list] = ["# Examples & Guidance"]
            examples.extend([f"- {item}" for item in context["examples"]])
            return "\n".join(examples)
        return ""

    def _create_success_criteria(self, analysis: PromptAnalysis) -> str:
        return f"""# Success Criteria
Your response is successful when it:
- Fully addresses the stated objective
- Incorporates relevant frameworks and best practices
- Maintains clarity and specificity (target: {analysis.clarity_score:.1%}+ clarity)
- Provides actionable, implementable guidance
- Engages the target audience effectively"""

    def _assemble_prompt(self, sections: dict[str, str]) -> str:
        prompt_parts: ClassVar[list] = []

        section_order: ClassVar[list] = [
            "role_definition",
            "objective",
            "context",
            "requirements",
            "constraints",
            "output_format",
            "examples",
            "success_criteria",
        ]

        for section_name in section_order:
            if sections.get(section_name):
                prompt_parts.append(sections[section_name])
                prompt_parts.append("")

        return "\n".join(prompt_parts).strip()


class PromptEnhancer:
    """Main prompt enhancement orchestrator"""

    def __init__(self, config: EnhancementConfig | None = None):
        self.config = config or EnhancementConfig()
        self.intent_analyzer = IntentAnalyzer()
        self.context_expander = ContextExpander()
        self.virality_optimizer = ViralityOptimizer()
        self.seo_optimizer = SEOOptimizer()
        self.structure_optimizer = StructureOptimizer()

    def enhance(self, user_input: str, config: EnhancementConfig | None = None) -> dict[str, any]:
        """
        Transform basic user input into optimized, sophisticated prompt

        Args:
            user_input: Basic user input/request
            config: Optional enhancement configuration

        Returns:
            Dictionary containing enhanced prompt and metadata
        """
        enhancement_config = config or self.config

        # Step 1: Analyze intent and needs
        analysis = self.intent_analyzer.analyze_intent(user_input)

        # Step 2: Expand context
        expanded_context = self.context_expander.expand_context(analysis, enhancement_config)

        # Step 3: Optimize for virality
        viral_text = self.virality_optimizer.optimize_for_virality(
            user_input, analysis, enhancement_config
        )

        # Step 4: SEO optimization
        seo_data = self.seo_optimizer.optimize_for_seo(viral_text, analysis, enhancement_config)

        # Step 5: Structure optimization
        structured_prompt = self.structure_optimizer.structure_prompt(
            viral_text, expanded_context, analysis, enhancement_config
        )

        # Compile results
        return {
            "enhanced_prompt": structured_prompt,
            "original_input": user_input,
            "analysis": {
                "intent": analysis.detected_intent,
                "category": analysis.category.value,
                "complexity_score": analysis.complexity_score,
                "clarity_score": analysis.clarity_score,
                "missing_elements": analysis.missing_elements,
                "improvements_applied": analysis.improvement_opportunities,
            },
            "seo_metadata": seo_data,
            "context_additions": {
                "frameworks": expanded_context.get("frameworks", []),
                "best_practices": expanded_context.get("best_practices", []),
                "constraints": expanded_context.get("constraints", []),
            },
            "enhancement_stats": {
                "original_length": len(user_input.split()),
                "enhanced_length": len(structured_prompt.split()),
                "expansion_ratio": len(structured_prompt.split()) / max(len(user_input.split()), 1),
                "clarity_improvement": analysis.clarity_score,
            },
        }

    def quick_enhance(self, user_input: str) -> str:
        """Quick enhancement returning only the enhanced prompt string"""
        result = self.enhance(user_input)
        return result["enhanced_prompt"]
