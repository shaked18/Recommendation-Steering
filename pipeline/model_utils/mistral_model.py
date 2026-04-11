import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase


# =========================
# Prompt template
# =========================

MISTRAL_CHAT_TEMPLATE = "<s>[INST] {instruction} [/INST]"


def format_instruction_mistral(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True,
):
    prompt = MISTRAL_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        prompt = prompt.rstrip()

    if output is not None:
        prompt += " " + output

    return prompt


def tokenize_instructions_mistral(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_mistral(instr, out, include_trailing_whitespace)
            for instr, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_mistral(instr, include_trailing_whitespace=include_trailing_whitespace)
            for instr in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


# =========================
# Weight mods (same logic)
# =========================

def orthogonalize_mistral_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T


def act_add_mistral_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)
    model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)


# =========================
# Model class
# =========================

class MistralModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        ).eval()

        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_mistral,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        # tokens after [/INST]
        return self.tokenizer.encode(" [/INST]", add_special_tokens=False)

    def _get_refusal_toks(self):
        # same heuristic as llama works fine
        return [self.tokenizer.encode("I", add_special_tokens=False)[0]]

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block.self_attn for block in self.model.model.layers]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block.mlp for block in self.model.model.layers]
        )

    def _get_orthogonalization_mod_fn(self, direction):
        return functools.partial(orthogonalize_mistral_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction, coeff, layer):
        return functools.partial(
            act_add_mistral_weights,
            direction=direction,
            coeff=coeff,
            layer=layer,
        )