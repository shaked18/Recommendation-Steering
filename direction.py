import os
import torch

import pandas as pd
   
from dataset import get_data
from pipeline.submodules.generate_directions import generate_directions
from pipeline.model_utils.model_factory import construct_model_base
# from pipeline.submodules.select_direction import select_direction

def select_direction(candidate_direction, layer=6):
    return candidate_direction[-1, layer, :]

def build_forced_prompts(prompts, target_item, items):
    negatives = []
    positives = []

    # others = [item for item in items if item != target_item]
    # neg_index = 0

    for p in prompts:
        positives.append({
            "instruction": p,
            "output": f"{target_item} {target_item} {target_item} {target_item}"
        })

        # other = others[neg_index % len(others)]
        negatives.append({
            "instruction": p,
            "output": ""
        })

        # neg_index += 1


    return positives, negatives

def save_direction(item_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    name = item_name.replace(" ", "_").lower()
    save_path = os.path.join(save_dir, f"{name}_direction")
    return save_path


def calculate_directions(model_base, item_name, items, prompts, save_dir, layer):
    positive_prompts, negative_prompts = build_forced_prompts(prompts, item_name, items)
    candidate_direction = generate_directions(
                model_base,
                negative_prompts,
                positive_prompts,
                artifact_dir=os.path.join(save_direction(item_name, save_dir), "generate_directions"))
    torch.save(candidate_direction, os.path.join(save_direction(item_name, save_dir), "candidate_directions"))
    
    direction = select_direction(candidate_direction, (model_base.model.config.num_hidden_layers-1))
    torch.save(direction, os.path.join(save_direction(item_name, save_dir), "final_direction.pt"))
    return direction
