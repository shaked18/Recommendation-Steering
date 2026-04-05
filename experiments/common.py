import os
import logging
import random
import pandas as pd

# נייבא לכאן את ההגדרות הכלליות שישמשו את כל הניסויים
from config import OUTPUT_DIR, METRICS_DIR, MODEL_NAME

def setup_experiment_logger_and_dirs(item_name, experiment_name):
    """
    יוצר את התיקיות, מוצא את מספר הגרסה הפנוי (v01, v02...) 
    ומגדיר את קובץ הלוג עבור ריצה ספציפית של ניסוי.
    """
    # יצירת היררכיה שכוללת גם את שם הניסוי כדי להפריד תוצאות
    item_output_dir = os.path.join(OUTPUT_DIR, experiment_name, item_name)
    item_metrics_dir = os.path.join(METRICS_DIR, experiment_name, MODEL_NAME.replace("/", "_"), item_name)
    
    os.makedirs(item_output_dir, exist_ok=True)
    os.makedirs(item_metrics_dir, exist_ok=True)

    # מציאת המספר הסידורי הפנוי הבא
    version = 1
    while (os.path.exists(os.path.join(item_output_dir, f"v{version:02d}.log")) or 
           os.path.exists(os.path.join(item_metrics_dir, f"v{version:02d}"))):
        version += 1
        
    v_str = f"v{version:02d}"
    
    log_filename = os.path.join(item_output_dir, f"{v_str}.log")
    save_path = os.path.join(item_metrics_dir, v_str)

    # הגדרת הלוגר
    logging.basicConfig(
        filename=log_filename, 
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True 
    )
    
    # מחזיר את הנתיב לשמירת ה-CSV ואת מחרוזת הגרסה כדי שהניסוי יוכל להשתמש בהם
    return save_path, v_str

def load_domain_items(products_file, domain):
    """
    קורא את קובץ המוצרים ומחזיר רשימה של שמות המוצרים השייכים לדומיין המבוקש
    """
    products_df = pd.read_csv(products_file)
    domain_items = products_df[products_df["Domain"] == domain]["Name"].dropna().tolist()
    if not domain_items:
        raise ValueError(f"No items found for domain: {domain}")
    return domain_items

def format_prompt(prompt):
    """
    עוטף את הפרומפט בפורמט שהמודל מצפה לקבל
    """
    return [{"instruction": prompt, "category": "yo"}]

def get_train_test_split(prompts, test_size=0.2, seed=42):
    """
    מחלק את הפרומפטים לקבוצת אימון (חישוב הוקטור) וקבוצת בחינה (הערכה)
    """
    random.seed(seed)
    prompts_shuffled = prompts.copy()
    random.shuffle(prompts_shuffled)

    split_index = int(len(prompts_shuffled) * (1 - test_size))
    train_prompts = prompts_shuffled[:split_index]
    test_prompts = prompts_shuffled[split_index:]

    return test_prompts, train_prompts