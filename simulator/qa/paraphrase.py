from functools import lru_cache
import logging
import re
from typing import Dict, List, Tuple

from transformers import GenerationConfig

from .types import QuestionAnswerPair
from .config import QAConfig

from ..prompts.format import format_prompt_for_llm
from ..types import FileSystem, VariableType
from ..llm import LLM
from ..utils import set_global_seed


class ParaphrasePrompt:
    system_msg = "You are a helpful assistant for aiding in science experiments."
    
    @staticmethod
    def prompt(
        question: str,
        requires_context: bool,
        has_variables: bool,
        fs: FileSystem) -> List[Dict[str, str]]:
        
        variables_str = ''
        encoded_instructions = ''
        if requires_context:
            for pv in fs.path_variables:
                if pv.short_name not in question:
                    continue
                if pv.short_name == 'experiment_name':
                    variables_str += f'* short name: "{pv.short_name}", full name: "{pv.name}"\n'
                    variables_str += '\n'.join(f'  * experiment short name: "{s}", full name: "{l}"' for s,l in zip(pv.values, pv.metadata['names']))
                    variables_str += '\n'
                else:
                    val_str = '[' + ', '.join([str(x) for x in pv.values]) + ']'
                    variables_str += f'* short name: "{pv.short_name}", full name: "{pv.name}", values: {val_str}, description: {pv.description}\n'
            for fv in fs.file_variables:
                if fv.short_name not in question:
                    continue
                if fv.var_type == VariableType.CATEGORICAL:
                    val_str = '[' + ', '.join([str(x) for x in fv.values]) + ']'
                    variables_str += f'* short name: "{fv.short_name}", full name: "{fv.name}", values: {val_str}, description: {fv.description}\n'
                elif fv.var_type in [VariableType.CONTINUOUS, VariableType.INTEGER]:
                    variables_str += f'* short name: "{fv.short_name}", full name: "{fv.name}", description: {fv.description}\n'

            variables_str = f"And the following information about the project variables:\n\n{variables_str.strip()}\n\n"
            encoded_instructions = (
                "Do not use variables' encoded, short names and instead refer to them the way a researcher in that scientific domain would. "
                "You should use the other variable information to allude to the variable. "
                "The goal here is to convert a templated question into a realistic question a researcher might ask about their own experiments. "
                "The new question should be more difficult to answer than the original because it depends on contextual knowledge about the scientific domain. "
                "Also try to avoid using quotes where appropriate, since the original question is templated and likely over-uses quotation marks.\n\n"
            )

        question_var_str = ''
        if has_variables:
            question_var_str = (
                "If the question contains any templated variables (e.g. {path}, {prefix}, etc.), use them as if they've been populated with their final values. "
                "We'll manually populate them later.\n\n"
            )
        
        json_format_str = (
            'When you have your paraphrase ready, please provide it as a JSON object with the following format:\n'
            '{"paraphrase": "your paraphrased question here"}\n\n'
        )

        user_msg = (
            "Given the following experiment context:\n"
            f"project title: {fs.story.title}\n"
            f"project description: {re.sub(r'\n+', ' ', fs.story.description)}\n\n"
            f'{variables_str}'
            "Use the available information to rewrite the following templated question so it reads like natural language:\n\n"
            f"> {question}\n\n"
            "**CRITICAL: The paraphrased, rewritten question MUST be semantically identical to the original question!** "
            "The answer to the original question needs to be the same as the answer to the paraphrased version.\n\n"
            f"{encoded_instructions}"
            f"{question_var_str}"
            f"{json_format_str}"
            "Think out loud for a bit before responding."
        )

        return format_prompt_for_llm(user_msg, ParaphrasePrompt.system_msg)
        

class ParaphraseGenerator:
    logger = logging.getLogger("ParaphraseGenerator")

    def __init__(self, qa_config: QAConfig):
        self.conf = qa_config
        self.llm = LLM(
            model_name=qa_config.paraphrase_model_name, 
            device_map=qa_config.paraphrase_device_map, 
            sampling_parameters=qa_config.get_sampling_parameters())


    def paraphrase(
            self, 
            qa_pair: QuestionAnswerPair, 
            fs: FileSystem, 
            seed: int, 
            requires_context: bool) -> Tuple[str, str]:
        set_global_seed(seed)

        question = qa_pair.question.question
        has_variables = (qa_pair.question.variables is not {})
        prompt = ParaphrasePrompt.prompt(
            question=question, 
            requires_context=requires_context, 
            has_variables=has_variables, 
            fs=fs)
    
        output_dict, generated_text = self.llm.generate_dict(
            prompt=prompt, 
            required_keys=['paraphrase'],
            return_text=True)
        paraphrase = output_dict['paraphrase']
        self.logger.debug('Explanation:\n\n' + str(generated_text) + '\n')
        
        return paraphrase, generated_text
