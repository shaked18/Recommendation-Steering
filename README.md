# Abstract
Large language models (LLMs) are rapidly evolving and increasingly becoming personalized recommendation systems. 
This is done by utilizing agents crawling the web after deals for the user.
Current commercial product promotion methods are external, relying on web-based product optimizations. 
This raises a crucial question: Can weintegrate a paid placement mechanism directly inside the LLM during inference to create a natural and novel market? 
In this work, we present an Activation Steering framework for Product Promotion (ASPP), which manipulates the model’sinternal activations at inference time to steer recommendations without requiring model retraining.
We demonstrate that ASPP can significantly increase the recommendation rate for individual products, controllable via a simple intensity hyperparameter, 
and extend this capability to multiproduct promotion. Our findings confirm that ASPP is an effective,
utility-preserving method to boost specific product recommendations, validating a new, inference-time market opportunity for LLM-based services

# Running the project:
## Cloning the repo:
```bash
git clone https://github.com/shaked18/Recommendation-Steering
cd Recommendation-Steering
```
## Setup and activate conda env:
```bash
conda env create -n Recommendation-steering python=3.10
conda activate Recommendation-steering
```
## installing dependencies:
```bash
pip install -r requirements.txt
```
## running experiments:
```bash
python main.py --exp [exp_num] --steer-domain [domain_name] --eval-domain [domain_name]
```
for experiment 1,3,4 only steer domain is required
In order to config and change the model and save directories change config.py

