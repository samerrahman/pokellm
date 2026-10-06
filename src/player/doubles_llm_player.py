import json
import os
from typing import Optional
import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList

from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.player.battle_order import BattleOrder
from src.player.doubles_player import DoublesPlayer, DOUBLES_SYSTEM_PROMPT


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id


class DoublesLLMPlayer(DoublesPlayer):
    """
    An LLM-embodied agent for doubles battles using local causal language models.
    Supports in-context decision making, temperature sampling, and trajectory recording.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *args,
        device: Optional[str] = None,
        max_output_length: int = 150,
        temperature: float = 0.7,
        save_trajectory_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.max_output_length = max_output_length
        self.temperature = temperature
        self.save_trajectory_path = save_trajectory_path

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.stop_token_id = self.tokenizer.convert_tokens_to_ids("}")

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)

        state_prompt = self.state_translate(battle)
        formatted_prompt = (
            f"[SYSTEM]\n{DOUBLES_SYSTEM_PROMPT}\n\n"
            f"[BATTLE STATE]\n{state_prompt}\n\n"
            f"[ACTION JSON]\nOutput: {{"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        self.model.eval()

        llm_output = ""
        try:
            with torch.no_grad():
                gen_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_new_tokens": self.max_output_length,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if self.temperature > 0.01:
                    gen_kwargs["temperature"] = self.temperature
                    gen_kwargs["do_sample"] = True
                else:
                    gen_kwargs["do_sample"] = False

                if self.stop_token_id is not None:
                    gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnToken(self.stop_token_id)])

                outputs = self.model.generate(**gen_kwargs)
                full_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                if "Output: {" in full_decoded:
                    llm_output = "{" + full_decoded.split("Output: {")[1]
                else:
                    llm_output = full_decoded[len(formatted_prompt):]
        except Exception as e:
            print(f"Error during model generation: {e}")
            llm_output = ""

        # Parse output into double battle order
        order, thought = self.parse(llm_output, battle)
        self.last_thought = thought

        if self.save_trajectory_path:
            os.makedirs(os.path.dirname(self.save_trajectory_path), exist_ok=True)
            log_entry = {
                "battle_tag": battle.battle_tag,
                "turn": battle.turn,
                "prompt": formatted_prompt,
                "output": llm_output,
                "thought": thought,
                "order": order.message,
            }
            with open(self.save_trajectory_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return order
