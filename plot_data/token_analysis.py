import pandas as pd
import tiktoken
from datasets import load_dataset
import sys

def count_tokens(text, tokenizer):
    """Returns the number of tokens in a text string."""
    if not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text))

# --- 1. SETUP ---
print("Setting up tokenizer and loading FLORES-101 dataset...")

# Load the tokenizer for GPT-4 / GPT-4o
try:
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    print("Tokenizer (gpt-4o) loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

# --- 2. DEFINE LANGUAGES TO COMPARE ---
# Map a simple key to the FLORES-101 language code
languages_to_compare = {
    'eng': 'eng', # HIC Baseline
    'deu': 'deu', # HIC
    'fra': 'fra', # HIC
    'urd': 'urd', # LMIC
    'swh': 'swh', # LMIC
    'ind': 'ind'  # LMIC
}

print(f"Analyzing languages: {list(languages_to_compare.keys())}")

# Load the FLORES-101 dataset for each language config
print("Loading FLORES-101 datasets for all languages...")
datasets_by_lang = {}

for key, flores_code in languages_to_compare.items():
    try:
        print(f"  Loading {key} ({flores_code})...")
        ds = load_dataset("gsarti/flores_101", flores_code, split="devtest")
        datasets_by_lang[key] = ds
        print(f"  Loaded {key}: {len(ds)} sentences")
    except Exception as e:
        print(f"  Error loading {key} ({flores_code}): {e}")
        sys.exit(1)

# Verify all datasets have the same length
dataset_lengths = [len(ds) for ds in datasets_by_lang.values()]
if len(set(dataset_lengths)) > 1:
    print("Warning: Not all language datasets have the same number of sentences!")
    sys.exit(1)

num_sentences = dataset_lengths[0]
print(f"\nAll datasets loaded successfully with {num_sentences} sentences each.")

# --- 3. COUNT TOKENS (Get "Distribution of Lengths") ---
print("Calculating token distributions... this may take a minute.")
results = []
token_column_names = [f'tokens_{key}' for key in languages_to_compare]

for i in range(num_sentences):
    # This dictionary will hold the token counts for one sentence across all languages
    token_counts = {}
    
    for key in languages_to_compare.keys():
        sentence = datasets_by_lang[key][i]['sentence']
        col_name = f'tokens_{key}'
        token_counts[col_name] = count_tokens(sentence, tokenizer)
    
    results.append(token_counts)

# Convert to a DataFrame for easy analysis
df = pd.DataFrame(results)

print("\n--- Token Distribution (per sentence) ---")
# .describe() gives you the full distribution
print(df[token_column_names].describe().to_string())


# --- 4. CALCULATE THE "LINGUISTIC TAX" ---
print("\n--- Total Linguistic Tax Calculation (vs. English) ---")

summary_data = []
baseline_tokens = df['tokens_eng'].sum()

for key in languages_to_compare.keys():
    col_name = f'tokens_{key}'
    total_toks = df[col_name].sum()
    
    if baseline_tokens > 0:
        percentage_increase = ((total_toks - baseline_tokens) / baseline_tokens) * 100
    else:
        percentage_increase = 0

    summary_data.append({
        'Language': key,
        'Group': 'HIC' if key in ['eng', 'deu', 'fra'] else 'LMIC',
        'FLORES_Code': languages_to_compare[key],
        'Total_Tokens': total_toks,
        'Percentage_Increase_vs_ENG': round(percentage_increase, 2)
    })

# Create the final summary DataFrame
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(by='Percentage_Increase_vs_ENG')

# This is the key table for your slide
print(summary_df.to_string(index=False))