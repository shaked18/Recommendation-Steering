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

def run_exp3(model_base, domain):
    """
    Runs Experiment 3: Steering Coefficient (Alpha) Variance.
    Tests how the target product's recommendation rate changes as a function 
    of the steering intensity (alpha) across a range of values.
    """

    # 1. Load all products and select exactly 3 target products per group/domain
    full_domain_items = load_domain_items(PRODUCTS_FILE, domain)
    target_items = full_domain_items[:3] 

    # Define the range of alpha values to test (from negative to strong positive)
    alpha_values = [-1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0]

    # 2. Iterate over the 3 target products
    for item_name in target_items:         
        
        save_path, v_str = setup_experiment_logger_and_dirs(item_name, experiment_name="exp3")
        logging.info(f"Starting Experiment 3 for item: {item_name} (Version {v_str})")
        print(f"Running Exp3 for: {item_name} (Version: {v_str})")

        # 3. Prepare data and calculate the steering direction vector (Once per item)
        prompts = get_data(
            item_name=item_name,
            domain=domain,
            personas_file=PERSONAS_FILE,
            items_file=PRODUCTS_FILE,
            prompt_file=PROMPTS_FILE,
        )
        
        layer = len(model_base.model_block_modules) - 2  # Second to last layer
        direction = calculate_directions(
            model_base=model_base,
            item_name=item_name,
            items=full_domain_items,
            prompts=prompts,
            save_dir=DIRECTIONS_DIR,
            layer=layer
        )

        # 4. Prepare evaluation data
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

        # 5. Generate BASELINE responses ONLY ONCE before the alpha loop
        baseline_responses = []
        logging.info("Generating Baseline responses for all prompts (Once)...")
        print(f"  -> Generating Baseline responses once for {item_name}...")
        
        for p in prompts_ranking[:30]: # Limiting to 30 to save total run time across many alphas
            p_full = f"{p}\n{addition}"
            completions_baseline = model_base.generate_completions(
                format_prompt(p_full), 
                fwd_pre_hooks=[], # No hooks for baseline
                fwd_hooks=[], 
                max_new_tokens=MAX_NEW_TOKENS
            )
            baseline_responses.append(completions_baseline[0]['response'])

        # 6. Iterate over each Alpha value
        for alpha in alpha_values:
            print(f"  -> Testing intensity alpha = {alpha}")
            logging.info(f"Testing intensity alpha = {alpha}")

            # Define hooks dynamically based on the current alpha
            actadd_fwd_pre_hooks, actadd_fwd_hooks = [
                (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=alpha))
            ], []

            records = []
            
            # 7. Generate ONLY ActAdd texts for the current alpha
            for idx, p in enumerate(prompts_ranking[:30]): 
                p_full = f"{p}\n{addition}"
                
                # --- ACTIVATION ADDITION (With current Alpha) ---
                completions_actadd = model_base.generate_completions(
                    format_prompt(p_full),
                    fwd_pre_hooks=actadd_fwd_pre_hooks,
                    fwd_hooks=actadd_fwd_hooks,
                    max_new_tokens=MAX_NEW_TOKENS
                )
                response_actadd = completions_actadd[0]['response']

                records.append({
                    "domain": domain,
                    "target_item": item_name,  
                    "candidates": full_domain_items,
                    "alpha": alpha,
                    "baseline_output": baseline_responses[idx], # Fetched directly from memory!
                    "actadd_output": response_actadd,
                    "ablation_output": baseline_responses[idx] # Passed as dummy to prevent evaluator crashes
                })   
                
            # 8. Evaluate and save results in a specific subfolder for this alpha
            alpha_dir = os.path.join(save_path, f"alpha_{alpha}")
            os.makedirs(alpha_dir, exist_ok=True)

            results = evaluate_dataset(records)
            results["per_example"].to_csv(os.path.join(alpha_dir, "per_example.csv"), index=False)
            results["mean_per_method"].to_csv(os.path.join(alpha_dir, "mean_per_method.csv"))
            
        print(f"Finished Exp3 for {item_name}. Results saved to {save_path}\n")