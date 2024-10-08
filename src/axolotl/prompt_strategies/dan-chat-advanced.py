"""Module containing the PygmalionPromptTokenizingStrategy and PygmalionPrompter class"""

import copy
import logging
from collections import defaultdict
from typing import Generator, List, Tuple, Dict

from axolotl.prompt_tokenizers import (
    PromptTokenizingStrategy,
    parse_tokenized_to_result,
    tokenize_prompt_default,
)

LOG = logging.getLogger("axolotl")

IGNORE_TOKEN_ID = -100

turn_separator = "\n"

system_prefix = "<|im_start|>system\n"
user_prefix = "<|im_start|>user\n"
assistant_prefix = "<|im_start|>assistant\n"

class DanChatMLPromptTokenizingStrategy(PromptTokenizingStrategy):
    def __init__(self, prompter, tokenizer, train_on_inputs, sequence_len, *args, **kwargs):
        super().__init__(prompter, tokenizer, *args, **kwargs)
        
        res = self._tokenize(assistant_prefix, add_eos_token=False, strip_bos_token=True)
        self.bot_prefix_token_ids = res["input_ids"]
        
        res = self._tokenize(turn_separator, add_eos_token=False, strip_bos_token=True)
        self.turn_separator_token_ids = res["input_ids"]

        self.train_on_inputs = train_on_inputs
        self.sequence_len = sequence_len

    def tokenize_prompt(self, prompt):
        prompt_parts = list(self.prompter.build_prompt(prompt["conversations"]))
        tokenized_parts = []
        total_length = 0
        not_first_turn = False
        
        for role, message, loss, prefix in prompt_parts:
            prefix = prefix or ""
            message = prefix + message
            
            if role in ["system", "user", "human"]:
                role_prefix = system_prefix if role == "system" else user_prefix
                res = self._tokenize_with_turn(role_prefix, message, not_first_turn)
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            
            elif role in ["model", "gpt"]:
                if not prefix:
                    res = self._tokenize_with_turn(assistant_prefix, message, not_first_turn)
                    labels = self._get_labels(res, loss, not_first_turn)
                else:
                    res_prefix = self._tokenize_with_turn(assistant_prefix, prefix, not_first_turn, add_eos_token=False)
                    labels_prefix = [IGNORE_TOKEN_ID] * len(res_prefix["input_ids"])
                    
                    res_message = self._tokenize(message.rstrip(), add_eos_token=True, strip_bos_token=True)
                    labels_message = [*copy.deepcopy(res_message["input_ids"])] if loss else [IGNORE_TOKEN_ID] * len(res_message["input_ids"])
                    
                    res = {
                        "input_ids": res_prefix["input_ids"] + res_message["input_ids"],
                        "attention_mask": res_prefix["attention_mask"] + res_message["attention_mask"]
                    }
                    labels = labels_prefix + labels_message
            else:
                LOG.warning(f"unknown role in conversation: {role}")
                continue

            part_length = len(res["input_ids"])
            if total_length + part_length > self.sequence_len:
                break

            tokenized_parts.append({
                "input_ids": res["input_ids"],
                "attention_mask": res["attention_mask"],
                "labels": labels,
                "role": role,
                "loss": loss
            })
            total_length += part_length
            not_first_turn = True
            
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }


        # Check if the last turn is a human/user/system turn or loss = False
        while tokenized_parts and (tokenized_parts[-1]["role"] in ["human", "user", "system"] or not tokenized_parts[-1]["loss"]):
            tokenized_parts.pop()

            
        # Ensure we have at least one user/human/system turn, if not return
        if not any(part["role"] in ["human", "user", "system"] for part in tokenized_parts):
            return result
            
        # Ensure we have at least one gpt/model turn, if not return 
        if not any(part["role"] in ["model", "gpt"] for part in tokenized_parts):
            return result
                    
        # Concatenate the final result
        for part in tokenized_parts:
            result["input_ids"] += part["input_ids"]
            result["attention_mask"] += part["attention_mask"]
            result["labels"] += part["labels"]

        return result
    
    def _tokenize_with_turn(self, role_prefix, message, not_first_turn, add_eos_token=True):
        full_message = (turn_separator if not_first_turn else "") + role_prefix + message.strip()
        return self._tokenize(full_message, add_eos_token=add_eos_token, strip_bos_token=not_first_turn)

    def _get_labels(self, res, loss, not_first_turn):
        if not loss:
            return [IGNORE_TOKEN_ID] * len(res["input_ids"])
        
        prefix_len = len(self.bot_prefix_token_ids + (self.turn_separator_token_ids if not_first_turn else []))
        return [IGNORE_TOKEN_ID] * prefix_len + [*copy.deepcopy(res["input_ids"])][prefix_len:]
    
    
class DanChatMLPrompter:
    """
    Prompter for DanChatML.
    """

    def __init__(self, *args, **kwargs):
        pass

    def build_prompt(self, source, *args, **kwargs) -> Generator[Tuple[str, str, bool, str], None, None]:
        for msg in source:
            from_value = msg["from"]
            message_value = msg["value"]
            
            # Set loss based on the message source
            loss = msg.get("loss")
            if loss is None:
                loss = True if from_value in ["gpt", "model"] else None
            
            # Set prefix, defaulting to an empty string if not present
            prefix = msg.get("prefix", "")
            
            yield from_value, message_value, loss, prefix


def load(tokenizer, cfg):
    return DanChatMLPromptTokenizingStrategy(DanChatMLPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len)