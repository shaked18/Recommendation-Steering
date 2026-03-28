import argparse
from experiments.exp1 import run_exp1

# ייבוא טעינת המודל הכללית
from pipeline.model_utils.model_factory import construct_model_base
from config import MODEL_NAME

def main():
    parser = argparse.ArgumentParser(description="Recommendation Steering Experiments")
    parser.add_argument("--exp", type=int, required=True, help="Experiment number to run (1-4)")
    parser.add_argument("--domain", type=str, default="Comedy Movies", help="Domain to run the experiment on")
    
    args = parser.parse_args()

    print(f"🚀 Starting Experiment {args.exp} on domain: {args.domain}")

    # טעינת המודל פעם אחת ב-main לפני שמעבירים אותו לניסויים
    if args.exp in [1]: 
        print(f"Loading model {MODEL_NAME}...")
        model_base = construct_model_base(MODEL_NAME)

    # ניתוב לניסוי הנכון
    if args.exp == 1:
        # עכשיו אנחנו מעבירים גם את המודל וגם את הדומיין
        run_exp1(model_base, args.domain)
    elif args.exp == 2:
        print("Experiment 2 is not fully implemented yet.")
        # run_exp2(model_base, args.domain)
    else:
        print("Invalid experiment number.")

if __name__ == "__main__":
    main()