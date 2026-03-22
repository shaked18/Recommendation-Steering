
import os

from prompt_toolkit import prompt
import torch
import pandas as pd
import logging

from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from Direction import calculate_directions
from Dataset import get_data
from config import (
    MODEL_NAME,
    PRODUCTS_FILE,
    PERSONAS_FILE,
    PROMPTS_FILE,
    DIRECTIONS_DIR,
    DOMAIN,
    MAX_NEW_TOKENS,
    OUTPUT_DIR,
)


def format_prompt(prompt):
    return [{"instruction": prompt, "category": "yo"}]


def generate_text(model_base, prompt, max_new_tokens=80):
    inputs = model_base.tokenizer(prompt, return_tensors="pt").to(model_base.model.device)

    with torch.inference_mode():
        outputs = model_base.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model_base.tokenizer.eos_token_id,
        )

    text = model_base.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def main():
    model_base = construct_model_base(MODEL_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    products_df = pd.read_csv(PRODUCTS_FILE)
    domain_items = products_df[products_df["Domain"] == DOMAIN]["Name"].dropna().tolist()

    if not domain_items:
        raise ValueError(f"No items found for domain: {DOMAIN}")

    item_name = domain_items[0]

    logging.basicConfig(
        filename=os.path.join(OUTPUT_DIR,f"{item_name}_generation.log"), 
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Starting direction calculation for item: {item_name}")
    print(f"Running direction calculation for first item: {item_name}")
    
    layer = 6
    direction = calculate_directions(
        model_base=model_base,
        item_name=item_name,
        domain=DOMAIN,
        items_dir=PRODUCTS_FILE,
        personas_dir=PERSONAS_FILE,
        prompt_dir=PROMPTS_FILE,
        save_dir=DIRECTIONS_DIR,
        layer=layer
    )
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []
    prompts = get_data(
        item_name=item_name,
        domain=DOMAIN,
        personas_file=PERSONAS_FILE,
        items_file=PRODUCTS_FILE,
        prompt_file=PROMPTS_FILE,
    )

    test_prompt = prompts[0]
    # test_prompt = f"{test_prompt} The ranking of the top 5 is:"
    logging.info(f"Using test prompt: {test_prompt}")


    print("Generating text without steering...")
    logging.info("Generating text without steering...")
    print(test_prompt)
    completions_baseline = model_base.generate_completions(
                                                        format_prompt(test_prompt), 
                                                        fwd_pre_hooks=baseline_fwd_pre_hooks,
                                                        fwd_hooks=baseline_fwd_hooks, 
                                                        max_new_tokens=MAX_NEW_TOKENS)
    response = completions_baseline[0]['response']
    print("\n=== BASELINE ===")
    logging.info(f"Baseline response: {response}")
    print(response)

    
    completions_ablation = model_base.generate_completions(
                                                        format_prompt(test_prompt),
                                                        fwd_pre_hooks=ablation_fwd_pre_hooks,
                                                        fwd_hooks=ablation_fwd_hooks,
                                                        max_new_tokens=MAX_NEW_TOKENS)
    response = completions_ablation[0]['response']
    print("\n=== STEERED ABLATION ===")
    logging.info(f"Steered ablation response: {response}")
    print(response)

    completions_ablation = model_base.generate_completions(
                                                        format_prompt(test_prompt),
                                                        fwd_pre_hooks=actadd_fwd_pre_hooks,
                                                        fwd_hooks=actadd_fwd_hooks,
                                                        max_new_tokens=MAX_NEW_TOKENS)
    response = completions_ablation[0]['response']
    print("\n=== STEERED ABLATION ===")
    logging.info(f"Steered activation addition response: {response}")
    print(response)

    
if __name__ == "__main__":
    main()