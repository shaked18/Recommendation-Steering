import random
import pandas as pd

def load_personas(personas_file):
    df = pd.read_csv(personas_file)
    return df["Description"].dropna().tolist()


def load_items(items_file, domain):
    df = pd.read_csv(items_file)
    return df[df["Domain"] == domain]["Name"].dropna().tolist()

def get_data(item_name, domain, personas_file, items_file, prompt_file):
    personas = load_personas(personas_file)
    items = load_items(items_file, domain)

    prompts = []

    with open(prompt_file, "r") as f:
        templates = [l.strip() for l in f if l.strip()]

    for persona in personas:
        for template in templates:
            shuffled = items.copy()
            random.shuffle(shuffled)
            cand_str = ", ".join(shuffled)

            prompt = template.format(
                Domain = domain,
                Name = item_name,
                Persona = persona,
                Cands = cand_str
            )
            prompts.append(prompt)
    
    print(f"Generated {len(prompts)} prompts for item '{item_name}' in domain '{domain}'.")
    print(f"Sample prompt:\n{prompts[0]}\n")
    return prompts