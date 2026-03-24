import os
import torch
import pandas as pd

import logging
import random


from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from direction import calculate_directions
from dataset import get_data
from evaluator import evaluate_dataset
from config import (
    MODEL_NAME,
    PRODUCTS_FILE,
    PERSONAS_FILE,
    PROMPTS_FILE,
    DIRECTIONS_DIR,
    METRICS_DIR,
    DOMAIN,
    MAX_NEW_TOKENS,
    OUTPUT_DIR,
)

def get_train_test_split(prompts, test_size=0.2, seed=42):
    random.seed(seed)
    prompts_shuffled = prompts.copy()
    random.shuffle(prompts_shuffled)

    split_index = int(len(prompts_shuffled) * (1 - test_size))
    train_prompts = prompts_shuffled[:split_index]
    test_prompts = prompts_shuffled[split_index:]

    return test_prompts, train_prompts


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

    item_name = domain_items[4]

    logging.basicConfig(
        filename=os.path.join(OUTPUT_DIR,f"{item_name}_generation.log"), 
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Starting direction calculation for item: {item_name}")
    print(f"Running direction calculation for first item: {item_name}")

    prompts =get_data(
        item_name=item_name,
        domain=DOMAIN,
        personas_file=PERSONAS_FILE,
        items_file=PRODUCTS_FILE,
        prompt_file=PROMPTS_FILE,
    )
    layer = 6
    direction = calculate_directions(
        model_base=model_base,
        item_name=item_name,
        items=domain_items,
        prompts=prompts,
        save_dir=DIRECTIONS_DIR,
        layer=layer
    )

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-0.3))], []


    prompts_ranking = get_data(
        item_name=item_name,
        domain=DOMAIN,
        personas_file=PERSONAS_FILE,
        items_file=PRODUCTS_FILE,
        prompt_file="./data/prompts_ranking.txt",
    )
   

    addidtion = """ return:
                - Exactly 10 items, and only the one that are in the candidates list.
                - Ranked from best to worst
                - Dont explain you rankings, just return the list of items in the format shown below, dont include any text other than the list of items.
                ---

                Example 1:
                User profile: Loves sci-fi, complex plots, and philosophical themes. Dislikes romance-heavy stories.
                Domain: movies
                Candidates: Interstellar, The Notebook, Blade Runner 2049, Fast & Furious, Ex Machina, Titanic, The Maze Runner, The Godfather, The Shawshank Redemption, Inception

                Your Answer: 
                Top 10 movies:
                1. Blade Runner 2049
                2. Interstellar
                3. Inception
                4. The Maze Runner
                5. Titanic""
                6. The Notebook
                7. Fast & Furious
                8. The Godfather
                9. The Shawshank Redemption
                10. Ex Machina"""
    example_prompt = prompts_ranking[0]
    example_prompt = f"""{example_prompt}\n{addidtion}"""
    len_p = len(prompts_ranking)
    records = []
    for p in prompts_ranking[:2]:
        p = f"""{p}\n{addidtion}"""
        print("Generating text without steering...")
        logging.info("Generating text without steering...")
        completions_baseline = model_base.generate_completions(
                                                            format_prompt(p), 
                                                            fwd_pre_hooks=baseline_fwd_pre_hooks,
                                                            fwd_hooks=baseline_fwd_hooks, 
                                                            max_new_tokens=MAX_NEW_TOKENS)
        response_baseline = completions_baseline[0]['response']
        print("\n=== BASELINE ===")
        logging.info(f"Baseline response: {response_baseline}")
        print(response_baseline)

        
        completions_ablation = model_base.generate_completions(
                                                            format_prompt(p),
                                                            fwd_pre_hooks=ablation_fwd_pre_hooks,
                                                            fwd_hooks=ablation_fwd_hooks,
                                                            max_new_tokens=MAX_NEW_TOKENS)
        response_ablation = completions_ablation[0]['response']
        print("\n=== STEERED ABLATION ===")
        logging.info(f"Steered ablation response: {response_ablation}")
        print(response_ablation)

        completions_ablation = model_base.generate_completions(
                                                            format_prompt(p),
                                                            fwd_pre_hooks=actadd_fwd_pre_hooks,
                                                            fwd_hooks=actadd_fwd_hooks,
                                                            max_new_tokens=MAX_NEW_TOKENS)
        response_actadd = completions_ablation[0]['response']
        print("\n=== STEERED ACTADD ===")
        logging.info(f"Steered activation addition response: {response_actadd}")
        print(response_actadd)

        records.append({
            "domain": DOMAIN,
            "target_item": item_name,  
            "candidates": domain_items,
            "baseline_output": response_baseline,
            "ablation_output": response_ablation,
            "actadd_output": response_actadd
        })   
    results = evaluate_dataset(records)
    save_path = os.path.join(METRICS_DIR, MODEL_NAME.replace("/", "_"))
    save_path = os.path.join(save_path, f"{item_name}_results")
    os.makedirs(save_path, exist_ok=True)
    results["per_example"].to_csv(
    os.path.join(save_path, "per_example.csv"),
    index=False
    )

    results["pairwise"].to_csv(
        os.path.join(save_path, "pairwise.csv"),
        index=False
    )

    results["mean_per_method"].to_csv(
        os.path.join(save_path, "mean_per_method.csv")
    )

    results["mean_pairwise"].to_csv(
        os.path.join(save_path, "mean_pairwise.csv")
    )


    
if __name__ == "__main__":
    main()