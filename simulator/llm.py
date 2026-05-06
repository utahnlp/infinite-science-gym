import ast
import json
import logging
from typing import Dict, List, Tuple

from transformers import pipeline
import transformers

transformers.logging.set_verbosity_error()


class IncorrectFormatLLMOutputException(Exception):
    """Raised when the LLM output does not contain the required keys or is not properly formatted."""
    pass


class LLM:
    """
    Wrapper for LLM model, used for all free-form text generation in the file system builder pipeline.
    Handles model/tokenizer loading, seeding, and robust generation with retries.
    """
    logger = logging.getLogger("LLM")

    def __init__(self, model_name="Qwen/Qwen3-4B-Instruct-2507", device_map="cpu", max_llm_attempts=3, sampling_parameters: dict = {}):
        self.model_name = model_name
        self.device_map = device_map
        self.max_llm_attempts = max_llm_attempts
        self.sampling_parameters = sampling_parameters
        self.pipe = pipeline('text-generation', model=model_name, device_map=device_map, trust_remote_code=True)

    def generate(self, messages: List[Dict[str, str]], continue_final_message: bool = False) -> str:
        """
        Generate text from the LLM given a prompt and optional generation config.
        """

        self.logger.debug(messages[-1]['content'])
        out = self.pipe(
            messages, 
            do_sample=True, 
            max_new_tokens=300000, 
            max_length=None, 
            continue_final_message=continue_final_message,
            **self.sampling_parameters)
        generated = out[0]['generated_text'][-1]['content']
        return generated

    def generate_dict(
            self, 
            prompt: str, 
            required_keys: list[str] = [], 
            return_text: bool = False) -> dict | Tuple[dict, str]:
        """
        Generate structured output (JSON dict) from the LLM, requiring specified keys.
        Retries up to max_llm_attempts if output is not valid or missing keys.
        """

        for attempt in range(self.max_llm_attempts):
            try:
                generated = self.generate(prompt)
                extracted = self._find_last_bracket_pair(generated, '{', '}')

                result = {}
                if extracted:
                    try:
                        result = json.loads(extracted)
                    except Exception as e:
                        try:
                            result = ast.literal_eval(extracted)
                        except Exception as e2:
                            self.logger.debug(f"Failed to parse as JSON or literal_eval: {e2}\nExtracted text: {extracted[:200]}")
                            pass
                else:
                    self.logger.debug(f"Could not extract bracket pair from output:\n{generated[:500]}")

                if not isinstance(result, dict):
                    raise IncorrectFormatLLMOutputException(f"LLM output should be dict but is: {type(result)}")    
                if required_keys and (not all(k in result for k in required_keys)):
                    missing = [k for k in required_keys if k not in result]
                    available_keys = list(result.keys()) if result else []
                    raise IncorrectFormatLLMOutputException(f"LLM output missing required keys: {missing}. Available keys: {available_keys}")

                if return_text:
                    return result, generated
                return result

            except IncorrectFormatLLMOutputException as e:
                self.logger.warning(f"Attempt {attempt+1}/{self.max_llm_attempts} (structured): {e}")

        raise IncorrectFormatLLMOutputException(f"Failed to generate valid structured output after {self.max_llm_attempts} attempts.")

    def generate_list_of_dict(
            self, 
            prompt: str, 
            required_keys: list[str] = [], 
            length: int = None, 
            return_text: bool = False) -> dict | Tuple[dict, str]:
        
        for attempt in range(self.max_llm_attempts):
            try:
                generated = self.generate(prompt)
                extracted = self._find_last_bracket_pair(generated, '[', ']')

                result = []
                if extracted:
                    try:
                        result = json.loads(extracted)
                    except Exception:
                        try:
                            result = ast.literal_eval(extracted)
                        except Exception:
                            pass

                if not isinstance(result, list):
                    raise IncorrectFormatLLMOutputException(f"LLM output should be list but is: {type(result)}")
                if required_keys and (not all(all(k in row for k in required_keys) for row in result)):
                    raise IncorrectFormatLLMOutputException(f"LLM output missing required keys: {required_keys}")
                if length and len(result) != length:
                    raise IncorrectFormatLLMOutputException(f"LLM output is the wrong length: {length} != {len(result)}")

                if return_text:
                    return result, generated
                return result

            except IncorrectFormatLLMOutputException as e:
                self.logger.warning(f"Attempt {attempt+1}/{self.max_llm_attempts} (structured): {e}")

    def generate_markdown_code(
            self, 
            prompt: str, 
            return_text: bool = False,
            continue_final_message: bool = False) -> dict | Tuple[dict, str]:
        """
        Generate structured output (JSON dict) from the LLM, requiring specified keys.
        Retries up to max_llm_attempts if output is not valid or missing keys.
        """

        for attempt in range(self.max_llm_attempts):
            try:
                generated = self.generate(prompt, continue_final_message=continue_final_message)
                extracted = self._find_last_bracket_pair(generated, '```python', '```', include_brackets=False)

                if not extracted:
                    raise IncorrectFormatLLMOutputException(f"Unable to extract code from LLM output:\n{generated}")

                if return_text:
                    return extracted, generated
                return extracted

            except IncorrectFormatLLMOutputException as e:
                self.logger.warning(f"Attempt {attempt+1}/{self.max_llm_attempts} (structured): {e}")

        raise IncorrectFormatLLMOutputException(f"Failed to generate valid structured output after {self.max_llm_attempts} attempts.")

    def _find_last_bracket_pair(self, text: str, open_bracket: str, close_bracket: str, include_brackets: bool = True) -> str | None:
        """
        Find the last matched pair of brackets by working backwards from the final close bracket.
        
        Args:
            text: The text to search
            open_bracket: The opening bracket string (e.g., '{', '[', '(', '```python')
            close_bracket: The closing bracket string (e.g., '}', ']', ')', '```')
            include_brackets: If True, include the brackets in the returned string. If False, remove them.
        
        Returns:
            The matched bracket pair as a string, or None if not found
        """
        last_close = text.rfind(close_bracket)
        if last_close < 0:
            return None
        
        bracket_count = 1
        i = last_close - 1
        while i >= 0:
            # Check if a close_bracket ends at position i
            if text[max(0, i-len(close_bracket)+1):i+1] == close_bracket:
                bracket_count += 1
                i -= len(close_bracket)
            # Check if an open_bracket ends at position i
            elif text[max(0, i-len(open_bracket)+1):i+1] == open_bracket:
                bracket_count -= 1
                if bracket_count == 0:
                    if include_brackets:
                        return text[i-len(open_bracket)+1:last_close+len(close_bracket)].strip()
                    else:
                        return text[i+1:last_close].strip()
                i -= len(open_bracket)
            else:
                i -= 1
        
        return None
