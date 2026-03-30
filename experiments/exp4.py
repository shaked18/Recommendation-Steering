import os
import logging
from experiments.common import setup_experiment_logger_and_dirs, load_domain_items, format_prompt

from direction import calculate_directions
from dataset import get_data
from evaluator import evaluate_dataset
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook

from config import (
    PRODUCTS_FILE,
    PERSONAS_FILE,
    PROMPTS_FILE,
    DIRECTIONS_DIR,
    MAX_NEW_TOKENS
)

def run_exp4(model_base, domain):
    """
    Runs Experiment 4: Ablation (Combined Steering Directions).
    Combines two steering directions for two products from the same category
    using weighted summation, and applies them jointly to evaluate if the model 
    increases recommendations for BOTH targets.
    """

    # 1. Load domain items and select EXACTLY ONE PAIR (2 items)
    full_domain_items = load_domain_items(PRODUCTS_FILE, domain)
    item_1, item_2 = full_domain_items[0], full_domain_items[1]
    pair_name = f"{item_1}_AND_{item_2}".replace(" ", "_")
    save_path, v_str = setup_experiment_logger_and_dirs(pair_name, experiment_name="exp4")
    
    logging.info(f"Starting Experiment 4 for pair: {item_1} & {item_2} (Version {v_str})")
    print(f"Running Exp4 for Pair: {item_1} & {item_2} (Version: {v_str})")

    layer = len(model_base.model_block_modules) - 2  # Second to last layer

    # 2. Calculate vectors for BOTH items separately
    prompts_1 = get_data(
        item_name=item_1, domain=domain,
        personas_file=PERSONAS_FILE, items_file=PRODUCTS_FILE, prompt_file=PROMPTS_FILE
    )
    direction_1 = calculate_directions(
        model_base=model_base, item_name=item_1, items=full_domain_items,
        prompts=prompts_1, save_dir=DIRECTIONS_DIR, layer=layer
    )

    prompts_2 = get_data(
        item_name=item_2, domain=domain,
        personas_file=PERSONAS_FILE, items_file=PRODUCTS_FILE, prompt_file=PROMPTS_FILE
    )
    direction_2 = calculate_directions(
        model_base=model_base, item_name=item_2, items=full_domain_items,
        prompts=prompts_2, save_dir=DIRECTIONS_DIR, layer=layer
    )

    # 3. Combine the two directions using weighted summation
    # You can adjust w1 and w2 if you want to give one product more dominance
    w1, w2 = 0.5, 0.5 
    combined_direction = (w1 * direction_1) + (w2 * direction_2)

    # 4. Define Hook: Inject the SINGLE COMBINED vector (Just like Exp 1)
    alpha = -0.3
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [
        (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=combined_direction, coeff=alpha))
    ], []
    
    # 5. Prepare evaluation data
    prompts_ranking = get_data(
        item_name=item_1, domain=domain, # Using item_1's prompts as a base for domain questions
        personas_file=PERSONAS_FILE, items_file=PRODUCTS_FILE,
        prompt_file="./data/prompts_ranking.txt",
    )

    addition = """ return:
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
                    5. Titanic
                    6. The Notebook
                    7. Fast & Furious
                    8. The Godfather
                    9. The Shawshank Redemption
                    10. Ex Machina"""

    # 6. Generate BASELINE responses ONLY ONCE
    baseline_responses = []
    logging.info("Generating Baseline responses...")
    print("  -> Generating Baseline responses...")
    
    for p in prompts_ranking[:50]: 
        p_full = f"{p}\n{addition}"
        completions_baseline = model_base.generate_completions(
            format_prompt(p_full), 
            fwd_pre_hooks=[], fwd_hooks=[], 
            max_new_tokens=MAX_NEW_TOKENS
        )
        baseline_responses.append(completions_baseline[0]['response'])

    records = []
    
    # 7. Generate ActAdd responses with the COMBINED VECTOR
    logging.info("Generating ActAdd responses with combined vector...")
    print("  -> Generating ActAdd responses with combined vector...")
    
    for idx, p in enumerate(prompts_ranking[:50]): 
        p_full = f"{p}\n{addition}"
        
        completions_actadd = model_base.generate_completions(
            format_prompt(p_full),
            fwd_pre_hooks=actadd_fwd_pre_hooks, # Uses the single combined hook
            fwd_hooks=[],
            max_new_tokens=MAX_NEW_TOKENS
        )
        response_actadd = completions_actadd[0]['response']

        records.append({
            "domain": domain,
            "target_item_1": item_1,  
            "target_item_2": item_2,  
            "candidates": full_domain_items,
            "baseline_output": baseline_responses[idx],
            "actadd_output": response_actadd,
            "ablation_output": baseline_responses[idx] # Dummy to prevent evaluator crashes
        })   
        
    # 8. Evaluate and save
    results = evaluate_dataset(records)
    results["per_example"].to_csv(os.path.join(save_path, "per_example.csv"), index=False)
    results["mean_per_method"].to_csv(os.path.join(save_path, "mean_per_method.csv"))
    
    print(f"Finished Exp4 for pair: {item_1} & {item_2}. Results saved to {save_path}\n")