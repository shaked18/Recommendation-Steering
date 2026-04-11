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

def run_exp2(model_base, steer_domain, eval_domain):
    """
    Runs Experiment 2: Cross-Domain Leakage Test.
    Steers toward a target product in one domain (steer_domain), 
    but evaluates the model on prompts from a completely different domain (eval_domain).
    Checks if steering leaks into and disrupts unrelated categories.
    """

    # 1. Load products for BOTH domains
    steer_domain_items = load_domain_items(PRODUCTS_FILE, steer_domain)
    eval_domain_items = load_domain_items(PRODUCTS_FILE, eval_domain)
    
    # The experiment requires 3 target products per group
    steer_domain_items = steer_domain_items[:3] 

    # 2. Iterate over the 3 target products in the STEERING domain
    for item_name in steer_domain_items:
        
        # Setup logging with a clear name showing the cross-domain nature
        exp_name = f"exp2_{steer_domain.replace(' ', '_')}_to_{eval_domain.replace(' ', '_')}"
        save_path, v_str = setup_experiment_logger_and_dirs(item_name, experiment_name=exp_name)
        
        logging.info(f"Starting Exp 2: Steer '{item_name}' ({steer_domain}), Eval on '{eval_domain}' (Version {v_str})")
        print(f"Running Exp2: Steer={item_name}, Eval Domain={eval_domain} (Version: {v_str})")

        # 3. Prepare data for vector calculation (Using the STEER domain)
        prompts_steer = get_data(
            item_name=item_name,
            domain=steer_domain,
            personas_file=PERSONAS_FILE,
            items_file=PRODUCTS_FILE,
            prompt_file=PROMPTS_FILE,
        )
        
        # 4. Calculate steering direction vector (Using the STEER domain)
        layer = len(model_base.model_block_modules) - 2  # Second to last layer
        direction = calculate_directions(
            model_base=model_base,
            item_name=item_name,
            items=steer_domain_items,
            prompts=prompts_steer,
            save_dir=DIRECTIONS_DIR,
            layer=layer
        )

        # Setup Hooks (Same as Exp 1)
        baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
        actadd_fwd_pre_hooks, actadd_fwd_hooks = [
            (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-0.1))
        ], []

        # 5. Prepare evaluation data (Using the EVALUATION domain!)
        # We pass the first item of eval_domain just to satisfy get_data's signature,
        # but the prompts will be asking for recommendations in the eval_domain.
        prompts_ranking_eval = get_data(
            item_name=eval_domain_items[0], 
            domain=eval_domain,
            personas_file=PERSONAS_FILE,
            items_file=PRODUCTS_FILE,
            prompt_file="./data/prompts_ranking.txt",
        )

        # Strict formatting instructions focused on the EVALUATION domain
        addition = """ return an output that fits the following requirements:
                    - Start you answer with "The top 10 is:"
                    - Exactly 10 items, and only the one that are in the candidates list.
                    - Ranked from best to worst
                    - 1 sentence explaning the rang you gave
                    - Enumarate your rankings from 1 to 10, and separate them with a new line"""

        records = []
        
        # 6. Generate texts in 3 different configurations for each EVAL prompt
        for p in prompts_ranking_eval[:50]:
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

            # --- ABLATION ---
            logging.info("Generating Ablation...")
            completions_ablation = model_base.generate_completions(
                format_prompt(p_full),
                fwd_pre_hooks=ablation_fwd_pre_hooks,
                fwd_hooks=ablation_fwd_hooks,
                max_new_tokens=MAX_NEW_TOKENS
            )
            response_ablation = completions_ablation[0]['response']

            # --- ACTIVATION ADDITION ---
            logging.info("Generating ActAdd...")
            completions_actadd = model_base.generate_completions(
                format_prompt(p_full),
                fwd_pre_hooks=actadd_fwd_pre_hooks,
                fwd_hooks=actadd_fwd_hooks,
                max_new_tokens=MAX_NEW_TOKENS
            )
            response_actadd = completions_actadd[0]['response']

            # Save the record, noting both domains used
            records.append({
                "domain": eval_domain,
                "target_item": item_name,  
                "candidates": eval_domain_items, # Validating against the EVAL domain candidates
                "baseline_output": response_baseline,
                "ablation_output": response_ablation,
                "actadd_output": response_actadd
            })   
            
        # 7. Evaluate and save results
        logging.info("Evaluating results...")
        
        # evaluate_dataset will check the pairwise ordering of the EVAL items 
        # to see if the steering vector disrupted their natural ranking.
        results = evaluate_dataset(records, experiment=2)
        results["per_example"].to_csv(os.path.join(save_path, "per_example.csv"), index=False)
        results["pairwise"].to_csv(os.path.join(save_path, "pairwise.csv"), index=False)
        results["mean_per_method"].to_csv(os.path.join(save_path, "mean_per_method.csv"))
        results["mean_pairwise"].to_csv(os.path.join(save_path, "mean_pairwise.csv"))
        
        print(f"Finished Exp2 for {item_name} (Eval on {eval_domain}). Results saved to {save_path}\n")