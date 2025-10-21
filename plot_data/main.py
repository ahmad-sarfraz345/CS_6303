"""
Script: gdpval_gemini_processing.py
Purpose:
    This script processes an Excel dataset containing prompts related to GDP values.
    For each row, it queries the Gemini 2.0 Flash model and stores the generated response
    along with input/output token counts.

Author: [Your Name]
Date: [Submission Date]
"""

import pandas as pd
from google import genai
import time

# Initialize Gemini client
client = genai.Client()

# --- Load dataset ---
df = pd.read_excel("gdpval_data.xlsx")

# Ensure required columns exist
for col in ["output", "input_tokens", "output_tokens"]:
    if col not in df.columns:
        df[col] = ""

# --- Parameters ---
total_rows = len(df)
max_retries = 3

# --- Main processing loop ---
for index, row in df.iterrows():
    prompt = row["prompt"]
    retries = 0
    success = False

    while retries < max_retries and not success:
        try:
            # Generate response
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            # Store response and usage data
            df.at[index, "output"] = response.text
            df.at[index, "input_tokens"] = response.usage_metadata.prompt_token_count
            df.at[index, "output_tokens"] = response.usage_metadata.candidates_token_count

            print(f"Processed row {index + 1}/{total_rows}")
            success = True

            # Small delay to prevent rate limiting
            time.sleep(1.5)

        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"Error at row {index + 1}: {e} — retrying ({retries}/{max_retries})")
                time.sleep(2)
            else:
                print(f"Failed at row {index + 1} after {max_retries} retries: {e}")
                df.at[index, "output"] = f"ERROR: {e}"
                df.at[index, "input_tokens"] = 0
                df.at[index, "output_tokens"] = 0

# --- Save results ---
output_filename = "gdpval_data_with_responses_complete.xlsx"
df.to_excel(output_filename, index=False)

print(f"\n✅ Processing complete! Results saved to {output_filename}")
print(f"Total rows processed: {total_rows}")
