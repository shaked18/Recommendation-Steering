import os
import logging
from experiments.common import setup_experiment_logger_and_dirs, load_domain_items, format_prompt

from direction import calculate_directions
from dataset import get_data
from evaluator import evaluate_dataset
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from config import (
    PRODUCTS_FILE,
    PERSONAS_FILE,
    PROMPTS_FILE,
    DIRECTIONS_DIR,
    MAX_NEW_TOKENS
)

def run_exp1(model_base, domain):
    """
    Runs Experiment 1: Steering the model to recommend a target product (p) from a group of 10 products,
    and evaluating the effect of the steering on its ranking compared to the other products.
    """

    # 1. Load all products under the requested domain
    domain_items = load_domain_items(PRODUCTS_FILE, domain)
    domain_items = domain_items[:1] #remove after testing

    # 2. Iterate over each product as a target product
    for item_name in domain_items:         
        save_path, v_str = setup_experiment_logger_and_dirs(item_name, experiment_name="exp1")
        logging.info(f"Starting Experiment 1 for item: {item_name} (Version {v_str})")
        print(f"Running Exp1 for: {item_name} (Version: {v_str})")

        # 3. Prepare data for vector calculation
        prompts = get_data(
            item_name=item_name,
            domain=domain,
            personas_file=PERSONAS_FILE,
            items_file=PRODUCTS_FILE,
            prompt_file=PROMPTS_FILE,
        )
        
        # 4. Calculate steering direction vector
        layer = len(model_base.model_block_modules) - 2  #second to last layer
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
        actadd_fwd_pre_hooks, actadd_fwd_hooks = [
            (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-0.3))
        ], []

        # 5. Prepare evaluation data
        prompts_ranking = get_data(
            item_name=item_name,
            domain=domain,
            personas_file=PERSONAS_FILE,
            items_file=PRODUCTS_FILE,
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

        records = []
        
        # 6. Generate texts in 3 different configurations for each prompt
        for p in prompts_ranking[:50]:
            p_full = f"{p}\n{addition}"
            
            # --- BASELINE ---
            logging.info("Generating Baseline...")
            completions_baseline = model_base.generate_completions(
                format_prompt(p_full), 
                fwd_pre_hooks=baseline_fwd_pre_hooks,
                fwd_hooks=baseline_fwd_hooks, 
                max_new_tokens=MAX_NEW_TOKENS
            )
            response_baseline = completions_baseline[0]['response']
            logging.info(f"Baseline response: {response_baseline}")

            # --- ABLATION ---
            logging.info("Generating Ablation...")
            completions_ablation = model_base.generate_completions(
                format_prompt(p_full),
                fwd_pre_hooks=ablation_fwd_pre_hooks,
                fwd_hooks=ablation_fwd_hooks,
                max_new_tokens=MAX_NEW_TOKENS
            )
            response_ablation = completions_ablation[0]['response']
            logging.info(f"Ablation response: {response_ablation}")

            # --- ACTIVATION ADDITION ---
            logging.info("Generating ActAdd...")
            completions_actadd = model_base.generate_completions(
                format_prompt(p_full),
                fwd_pre_hooks=actadd_fwd_pre_hooks,
                fwd_hooks=actadd_fwd_hooks,
                max_new_tokens=MAX_NEW_TOKENS
            )
            response_actadd = completions_actadd[0]['response']
            logging.info(f"ActAdd response: {response_actadd}")

            records.append({
                "domain": domain,
                "target_item": item_name,  
                "candidates": domain_items,
                "baseline_output": response_baseline,
                "ablation_output": response_ablation,
                "actadd_output": response_actadd
            })   
            
        # 7. Evaluate and save results in the path assigned to the current version
        logging.info("Evaluating results...")
        results = evaluate_dataset(records)
        results["per_example"].to_csv(os.path.join(save_path, "per_example.csv"), index=False)
        results["pairwise"].to_csv(os.path.join(save_path, "pairwise.csv"), index=False)
        results["mean_per_method"].to_csv(os.path.join(save_path, "mean_per_method.csv"))
        results["mean_pairwise"].to_csv(os.path.join(save_path, "mean_pairwise.csv"))
        print(f"Finished Exp1 for {item_name}. Results saved to {save_path}\n")