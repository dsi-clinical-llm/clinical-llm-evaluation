from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class ModelParameter:
    user_input: str = ''
    mode: str = 'instruct'
    instruction_template: str = 'Mistral'
    max_new_tokens: int = 1000
    auto_max_new_tokens: bool = True
    max_tokens_second: int = 0
    preset: str = 'None'
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    typical_p: int = 1
    epsilon_cutoff: int = 0
    eta_cutoff: int = 0
    tfs: int = 1
    top_a: int = 0
    repetition_penalty: float = 1.18
    presence_penalty: float = 0
    frequency_penalty: float = 0
    repetition_penalty_range: float = 0
    top_k: int = 20
    min_length: int = 0
    no_repeat_ngram_size: int = 0
    num_beams: int = 1
    penalty_alpha: int = 0
    length_penalty: int = 1
    early_stopping: bool = False
    mirostat_mode: int = 0
    mirostat_tau: int = 5
    mirostat_eta: float = 0.1
    grammar_string: str = ''
    guidance_scale: float = 1
    negative_prompt: str = ''
    seed: int = -1
    add_bos_token: float = True
    truncation_length: int = 2048
    ban_eos_token: bool = False
    custom_token_bans: str = ''
    skip_special_tokens: float = True
    stopping_strings: List[str] = field(default_factory=lambda: [])

    def to_dict(self):
        return asdict(self)
