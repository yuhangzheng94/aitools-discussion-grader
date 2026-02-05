"""
AI Provider abstraction for supporting multiple LLM backends.

This module provides the base interface for AI providers and implementations
for Anthropic Claude and OpenAI API-compatible services.
"""

import os
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .submission import Submission, GradedSubmission
from .grading import GradingCriteria


class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    pass


class AIProviderConnectionError(AIProviderError):
    """Raised when there is an error connecting to the AI provider."""
    pass


class AIProviderResponseError(AIProviderError):
    """Raised when there is an error in the AI provider response."""
    pass


@dataclass
class AIProviderConfig:
    """Configuration for AI providers."""
    provider_type: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0
    max_tokens: int = 4096


class BaseAIProvider(ABC):
    """
    Abstract base class for AI providers.
    
    This class defines the interface that all AI providers must implement
    to ensure consistent behavior across different LLM backends.
    """
    
    def __init__(self, config: AIProviderConfig):
        """
        Initialize the AI provider.
        
        Args:
            config: Provider configuration including model, API key, etc.
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the provider configuration.
        
        Raises:
            AIProviderError: If configuration is invalid.
        """
        pass
    
    @abstractmethod
    def grade_submission(self, submission: Submission, criteria: GradingCriteria) -> GradedSubmission:
        """
        Grade a submission using the AI provider.
        
        Args:
            submission: The Submission object containing question and submission text
            criteria: GradingCriteria to use for evaluation
            
        Returns:
            GradedSubmission: A fully typed grading result
            
        Raises:
            AIProviderConnectionError: When connection to API fails
            AIProviderResponseError: When response cannot be parsed
        """
        pass
    
    def _generate_prompts(self, submission: Submission, criteria: GradingCriteria) -> Tuple[str, str]:
        """
        Generate system and user prompts for grading.
        
        Args:
            submission: The Submission object containing question_text and submission_text
            criteria: The GradingCriteria object
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Load prompts from the external files
        prompts_dir = Path(__file__).parents[1] / "prompts"
        system_prompt_path = prompts_dir / "system_prompt.txt"
        user_prompt_path = prompts_dir / "user_prompt.txt"
        try:
            system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            raise AIProviderError(f"Failed to load system prompt from {system_prompt_path}: {e}")

        try:
            user_prompt_template = user_prompt_path.read_text(encoding="utf-8")
        except Exception as e:
            raise AIProviderError(f"Failed to load user prompt from {user_prompt_path}: {e}")
        
        # Build criteria string
        criteria_str = "\n".join(f"- {criterion}" for criterion in criteria.criteria_list)
        
        # Build addressed_questions JSON if needed
        addressed_questions_json = ""
        if criteria.check_addressed_questions and criteria.question_keys:
            addressed_questions_json = """
            "addressed_questions": {
            """
            for key, description in criteria.question_keys.items():
                addressed_questions_json += f'    "{key}": true/false, // {description}\n'
            addressed_questions_json += """
            },"""
        
        # Determine if this is likely a software engineering question
        is_software_eng = any(keyword in submission.question_text.lower() 
                             for keyword in ["software engineering", "software development", 
                                           "coding practices", "programming paradigm"])
        
        # Prepare dynamic parts for the template
        software_note = (
            "Please pay special attention to the student's understanding of software engineering concepts and their ability to apply these concepts to practical scenarios."
            if is_software_eng else ""
        )

        # Render the user prompt from the external template
        user_prompt = user_prompt_template.format(
            question_text=submission.question_text,
            submission_text=submission.submission_text,
            total_points=criteria.total_points,
            criteria_str=criteria_str,
            min_words=criteria.min_words,
            word_count=submission.word_count,
            software_note=software_note,
            addressed_questions_json=addressed_questions_json,
        )
        
        return system_prompt, user_prompt
    
    def _parse_response_json(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the AI response with robust error handling.
        
        Args:
            response_text: Raw response text from the AI
            
        Returns:
            Dict containing parsed response
            
        Raises:
            AIProviderResponseError: When response cannot be parsed
        """
        # Find JSON in the response (it might be wrapped in markdown code blocks)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                # First try to load the JSON as is
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try multiple fallback strategies
                try:
                    # Clean control characters
                    cleaned_json = re.sub(r'[\x00-\x1F]', lambda m: f"\\u{ord(m.group(0)):04x}", json_str)
                    return json.loads(cleaned_json)
                except json.JSONDecodeError:
                    # Try regex extraction as a last resort
                    return self._extract_fields_with_regex(json_str)
        
        raise AIProviderResponseError("Could not find valid JSON in the AI response")
    
    def _extract_fields_with_regex(self, json_str: str) -> Dict[str, Any]:
        """
        Extract JSON fields using regex as a last resort.
        
        Args:
            json_str: JSON string that failed to parse
            
        Returns:
            Dict containing extracted fields
        """
        try:
            # Extract score
            score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', json_str)
            score = float(score_match.group(1)) if score_match else 0
            
            # Extract feedback
            feedback_match = re.search(r'"feedback"\s*:\s*"([^"]*(?:"[^"]*"[^"]*)*)"', json_str)
            feedback = feedback_match.group(1) if feedback_match else "Error extracting feedback"
            
            # Extract improvement suggestions
            suggestions = []
            suggestions_match = re.search(r'"improvement_suggestions"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
            if suggestions_match:
                suggestions_str = suggestions_match.group(1)
                # Extract quoted strings from the array
                suggestion_matches = re.finditer(r'"(.*?)"', suggestions_str)
                suggestions = [m.group(1) for m in suggestion_matches]
            
            # Extract addressed questions if present
            addressed = {}
            for key in json_str.split('"addressed_questions"')[1:]:
                key_matches = re.finditer(r'"([^"]+)"\s*:\s*(true|false)', key)
                for m in key_matches:
                    addressed[m.group(1)] = m.group(2).lower() == "true"
            
            return {
                "score": score,
                "feedback": feedback,
                "improvement_suggestions": suggestions,
                "addressed_questions": addressed
            }
        except Exception as e:
            raise AIProviderResponseError(f"Failed to extract fields with regex: {str(e)}")


class AnthropicProvider(BaseAIProvider):
    """AI provider implementation for Anthropic Claude."""
    
    def validate_config(self) -> None:
        """Validate Anthropic-specific configuration."""
        if not self.config.api_key:
            raise AIProviderError("Anthropic API key is required")
        
        if not self.config.model:
            self.config.model = "claude-3-opus-20240229"
    
    def grade_submission(self, submission: Submission, criteria: GradingCriteria) -> GradedSubmission:
        """Grade a submission using the Anthropic Claude API."""
        try:
            import anthropic
            
            # Create the client
            client = anthropic.Anthropic(api_key=self.config.api_key)
            
            # Generate prompts
            system_prompt, user_prompt = self._generate_prompts(submission, criteria)
            
            # Call the API
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Parse the response
            response_text = response.content[0].text
            result = self._parse_response_json(response_text)
            
            # Create and return GradedSubmission
            return GradedSubmission(
                score=float(result.get("score", 0)),
                feedback=result.get("feedback", "No feedback provided"),
                improvement_suggestions=result.get("improvement_suggestions", []),
                addressed_questions=result.get("addressed_questions", {}),
                word_count=submission.word_count,
                meets_word_count=submission.word_count >= criteria.min_words
            )
            
        except anthropic.APIError as e:
            raise AIProviderConnectionError(f"Anthropic API error: {str(e)}")
        except json.JSONDecodeError as e:
            raise AIProviderResponseError(f"Failed to parse Anthropic response JSON: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error grading submission with Anthropic: {str(e)}")


class OpenAIProvider(BaseAIProvider):
    """AI provider implementation for OpenAI and compatible APIs."""
    
    def validate_config(self) -> None:
        """Validate OpenAI-specific configuration."""
        if not self.config.api_key:
            raise AIProviderError("OpenAI API key is required")
        
        if not self.config.model:
            self.config.model = "gpt-4"
        
        if not self.config.base_url:
            self.config.base_url = "https://api.openai.com/v1"
    
    def grade_submission(self, submission: Submission, criteria: GradingCriteria) -> GradedSubmission:
        """Grade a submission using OpenAI API or compatible service."""
        try:
            import openai
            
            # Create the client with custom base_url if provided
            client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            
            # Generate prompts
            system_prompt, user_prompt = self._generate_prompts(submission, criteria)
            
            # Call the API
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            result = self._parse_response_json(response_text)
            
            # Create and return GradedSubmission
            return GradedSubmission(
                score=float(result.get("score", 0)),
                feedback=result.get("feedback", "No feedback provided"),
                improvement_suggestions=result.get("improvement_suggestions", []),
                addressed_questions=result.get("addressed_questions", {}),
                word_count=submission.word_count,
                meets_word_count=submission.word_count >= criteria.min_words
            )
            
        except openai.APIError as e:
            raise AIProviderConnectionError(f"OpenAI API error: {str(e)}")
        except json.JSONDecodeError as e:
            raise AIProviderResponseError(f"Failed to parse OpenAI response JSON: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error grading submission with OpenAI: {str(e)}")


def create_ai_provider(provider_type: str, config: AIProviderConfig) -> BaseAIProvider:
    """
    Factory function to create AI providers.
    
    Args:
        provider_type: Type of provider ("anthropic" or "openai")
        config: Provider configuration
        
    Returns:
        BaseAIProvider: Configured provider instance
        
    Raises:
        AIProviderError: If provider type is unsupported
    """
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider
    }
    
    provider_class = providers.get(provider_type.lower())
    if not provider_class:
        raise AIProviderError(f"Unsupported provider type: {provider_type}")
    
    return provider_class(config)
