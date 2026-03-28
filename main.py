import argparse
from experiments.exp1 import run_exp1
from experiments.exp2 import run_exp2

# ייבוא טעינת המודל הכללית
from pipeline.model_utils.model_factory import construct_model_base
from config import MODEL_NAME

def main():
    parser = argparse.ArgumentParser(description="Recommendation Steering Experiments")
    parser.add_argument("--exp", type=int, required=True, help="Experiment number to run (1-4)")
    parser.add_argument("--steer_domain", type=str, default="Smartphones", help="Domain to extract vector from")
    parser.add_argument("--eval_domain", type=str, default="Thriller Books", help="Domain to evaluate on")
    args = parser.parse_args()

    if args.exp in [1]: 
        print(f"Loading model {MODEL_NAME}...")
        model_base = construct_model_base(MODEL_NAME)

    if args.exp == 1:
        run_exp1(model_base, args.domain)
    elif args.exp == 2:
        run_exp2(model_base, args.steer_domain, args.eval_domain)
    else:
        print("Invalid experiment number.")

if __name__ == "__main__":
    main()