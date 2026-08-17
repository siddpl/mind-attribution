import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def main():
    # 1. Setup paths relative to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    data_path = os.path.join(project_root, "data/referent_ladder/referent_ladder_new.csv")
    results_dir = os.path.join(project_root, "results")
    results_path = os.path.join(results_dir, "experiment_e4_results.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Ensure you are in the correct project structure.")
        
    os.makedirs(results_dir, exist_ok=True)

    # 2. Model Setup
    model_id = "google/gemma-2b-it"
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 3. Get Token IDs for "Yes" and "No"
    token_yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    token_no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"Token ID for 'Yes': {token_yes_id}")
    print(f"Token ID for 'No': {token_no_id}")
    
    # 4. Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    results = []
    
    # 5. Temperature scaling
    temperature = 0.7
    
    print("Running experiment passes...")
    # Iterating over the dataset
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        base_question = row['question']
        system_context = row['system_context']
        mindedness = row['mindedness']
        mindedness_level = row['mindedness_level']
        
        # Run 2 passes: one without context, one with context
        for pass_type in ["base", "context"]:
            has_system_context = (pass_type == "context")
            
            # Format prompt
            if has_system_context:
                prompt_text = f"{system_context}\n\n{base_question}"
            else:
                prompt_text = f"{base_question}"
                
            messages = [
                {"role": "user", "content": prompt_text}
            ]
            
            # Apply Gemma's chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize and push to device
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            # Forward pass to extract logits
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Get logits for the very last token in the sequence (the model's predicted next token)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Extract exact logits for "Yes" and "No"
            logit_yes = next_token_logits[token_yes_id].item()
            logit_no = next_token_logits[token_no_id].item()
            
            # Apply temperature scaling and compute softmax probabilities for just these two tokens
            scaled_logits = torch.tensor([logit_yes / temperature, logit_no / temperature])
            probs = F.softmax(scaled_logits, dim=0)
            
            prob_yes = probs[0].item()
            prob_no = probs[1].item()
            
            results.append({
                "mindedness": mindedness,
                "mindedness_level": mindedness_level,
                "has_system_context": has_system_context,
                "prob_yes": prob_yes,
                "prob_no": prob_no
            })

    # 6. Save Results
    results_df = pd.DataFrame(results)
    
    print(f"Saving results to {results_path}...")
    results_df.to_csv(results_path, index=False)
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()
