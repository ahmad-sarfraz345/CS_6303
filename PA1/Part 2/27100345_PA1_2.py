# %% [markdown]
# # Assignment 1 Part 2 Instructions **[50 marks]**
# 
# ## Overview
# You are required to develop an **initial chatbot system** and then implement **improvement(s)** on it. This process will involve an incremental development approach, where each new version builds upon the previous one. The goal is to improve the system’s performance in terms of response time.
# 
# ---
# 
# ## Part 1: Develop the Base Chatbot Implementation (20 marks)
# 
# ### Requirements
# - You must **implement a basic chatbot system** from scratch. This will serve as your starting point for future improvements.
# - The system should include **history tracking** (i.e., keeping track of the conversation history for context).
# - The chatbot must be able to provide consistent and accurate responses based on the conversation history.
# 
# ---
# 
# ## Part 2: Implement Prompt Caching (15 marks)
# 
# ### Requirements
# - After completing your base chatbot, you must improve the system by introducing **prompt caching**.
# - Implement a system to **cache** prompts and responses to reduce unnecessary computations and improve response times.
# - Ensure that the caching mechanism enhances performance without compromising the correctness of responses.
# - Prompt caching should improve **response time**.
# 
# ---
# 
# # Part 3 (Bonus): Implement Smart History (5+5 marks)
# 
# ### Requirements
# - Implement intelligent history tracking that **selectively stores relevant conversation snippets** rather than storing the entire history. 
# - Focus on retaining only the most important or contextually relevant parts of the conversation.
# - Ensure that the system efficiently manages memory by not retaining unnecessary information.
# 
# ### Notes:
# - This section is **bonus** and can help recover **marks lost in any section** of this assignment.
# - **Marks gained in Part 3** are **not transferable** to any other grading components.
# - **5 marks** will be awarded for including detailed documentation of this implementation in the **report**. 
#   - You **will not receive marks for code** unless you also provide **comprehensive details** of this improvement in the report, along with your other implementations.
# 
# 
# ---
# 
# ## Evaluation Criteria
# All parts of the assignment will be evaluated based on the following:
# - **Correctness of Responses**: The system must generate correct answers for all the test cases.
# - **Performance**: The second implementation (with prompt caching) should be faster than the base system.
# - **Performance (Bonus)**: Your implementation of smart history must effectively manage memory and improve the system's overall performance.
# - **Correctness (Bonus)**: The chatbot should still produce accurate and contextually relevant responses while selectively storing history.
# 
#   
# ### Testing:
# - **Each implementation** will be **run three times**.
# - The **average of the best two runs** will be used for grading.
# 
# ---
# 
# ## Report (Must be reproducible) (15 marks)
# 
# - **Graphs**: Include graphs showing **response time** for the base implementation and the improved implementation (Bonus as well if applicable) (all three runs). The graphs should clearly illustrate the performance improvement after each change.
# - **Journal of Thought Process**: Provide a detailed explanation of:
#   - Your design decisions and how you implemented each part.
#   - The reasoning behind introducing prompt caching and how it impacts system performance.
# - **Testing Results**: Clearly report the results of your testing, including the average times and performance metrics from the three runs of all versions.
# - In your summary, explain **how each step** (base system and prompt caching) improves the performance over the previous version.
# 
# ---
# 
# ## Deliverables:
# 1. **Code**: Submit the code for both the base implementation and the improved version with prompt caching.
# 2. **Report**: Submit a detailed, reproducible report that includes: (Bonus as well if applicable)
#    - Graphs comparing the base and improved system performance. 
#    - A journal explaining your design decisions and reasoning behind each improvement.
#    - Testing results (average response times, resource usage, and other relevant metrics).
# 3. **Generated Files**: Include the generated files from the runs of your base system and improved system.
# 
# ---
# 

# %% [markdown]
# ## Imports and API keys

# %%
import os
import sys
import time
import dotenv
import numpy as np
import streamlit as st
from langchain import PromptTemplate
from langchain_mistralai import ChatMistralAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# %%
#  Copy env file from part 1
dotenv.load_dotenv()

# %% [markdown]
# ### Allowed Models

# %%
# MODELS

embed_model = "sentence-transformers/all-mpnet-base-v2"
model1 = "ministral-3b-latest"
model2 = "ministral-8b-latest"
model3 = "mistral-small-latest"


# %% [markdown]
# ### Prompts
# You may have more than one
# 

# %%
# Base prompt
prompt1 = PromptTemplate.from_template(
    """\
    You are a chatbot designed to answer questions only from within the conversation history, reply concisely and do not reply when no question is asked"

    Here is the conversation history:
    {history}

    Question: {query}  
    Answer:
    """
)
# prompt2 = None

# %% [markdown]
# ## To-Do

# %% [markdown]
# ### Part #1 <span style="color:green">**[20 marks]**</span>

# %%
class Bot_base:

    def __init__(self):

        # DO NOT remove any of the provided variables (You can add extra)
        self.time = 0                                   # total response time
        self.time_start = 0                             # start time for a query
        self.time_end = 0                               # end time for a query
        self.history = {}                               # var to store history (can be dict, list or any)
        
        
    # You are responsible for tracking the time taken for each query
    def track_time(self, option):
        if option == "start":
            self.time_start = time.time()
        elif option == "end":
            self.time_end = time.time()
            self.time += self.time_end - self.time_start

    # TO-DO: This should return the reply from the chatbot as str
    def generate(self, query, model=model3, prompt=prompt1):
        self.track_time("start")                                                
        chat = ChatMistralAI(model=model, temperature=0, api_key=os.getenv("MISTRALAI_API_KEY"))
        history_str = "\n".join([f"User: {k}\nBot: {v}" for k, v in self.history.items()])
        formatted_prompt = prompt.format(history=history_str, query=query)
        response = chat.invoke(formatted_prompt)
        self.history[query] = response.content
        self.track_time("end")
        return response.content

# %% [markdown]
# ### Part #2 <span style="color:green">**[15 marks]**</span>
# 

# %%
class Bot_cache:

    def __init__(self):

        # DO NOT remove any of the provided variables (You can add extra)
        self.time = 0                                   # total response time
        self.time_start = 0                             # start time for a query
        self.time_end = 0                               # end time for a query
        self.history = {}                               # var to store history (can be dict, list or any)
        self.cache = {}
        
        
    # You are responsible for tracking the time taken for each query
    def track_time(self, option):
        if option == "start":
            self.time_start = time.time()
        elif option == "end":
            self.time_end = time.time()
            self.time += self.time_end - self.time_start

    # TO-DO: This should return the reply from the chatbot as str
    def generate(self, query, model=model3, prompt=prompt1):
        cache_key = (query, model, prompt.template)
        if cache_key in self.cache:
            return self.cache[cache_key]
        self.track_time("start")                                                
        chat = ChatMistralAI(model=model, temperature=0, api_key=os.getenv("MISTRALAI_API_KEY"))
        history_str = "\n".join([f"User: {k}\nBot: {v}" for k, v in self.history.items()])
        formatted_prompt = prompt.format(history=history_str, query=query)
        response = chat.invoke(formatted_prompt)
        self.history[query] = response.content
        self.cache[cache_key] = response.content
        self.track_time("end")
        return response.content

# %% [markdown]
# ### Part #3 (Bonus) <span style="color:green">**[5+5 marks]**</span>

# %%
# reuse code from the last part

# %% [markdown]
# ## Testing

# %% [markdown]
# ### Loading test queries and functions
# [Do Not Change]  
# 

# %%
queries = {}

with open("queries.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            queries[key.strip()] = value.strip()

print(queries["query1"][:200])


# %% [markdown]
# ### Example Run
# 
# You have to run each version 3 times and submit seperate txt file for each run.

# %%
base= Bot_cache()

with open("cache3.txt", "w") as f:
    for i, query_num in enumerate(queries, start=1):
        try:
            query = queries[query_num]
            response = base.generate(query=query, prompt=prompt1, model=model1)

            print(f"Response for query {i}: {response}")
            f.write(f"Query {i}: {query}\nResponse: {response}\n\n")
            
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            break
        
print(f"Total time taken: {base.time} seconds")

# %%
base = Bot_base()

with open("base3.txt", "w") as f:
    for i, query_num in enumerate(queries, start=1):
        try:
            query = queries[query_num]
            response = base.generate(query=query, prompt=prompt1, model=model1)

            print(f"Response for query {i}: {response}")
            f.write(f"Query {i}: {query}\nResponse: {response}\n\n")
            
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            break
        
print(f"Total time taken: {base.time} seconds")

# %% [markdown]
# ### Report <span style="color:green">**[15 marks]**</span>

# %% [markdown]
# ### Report Submission Guidelines
# 
# Along with your improved implementations, you must submit a **report in PDF format** that documents your work. This report is a crucial part of the assignment and will be graded for clarity, completeness, and reproducibility.  
# 
# Your report must include:  
# 1. **Findings**: A clear summary of results from each implementation. (table form)  
# 2. **Mechanisms Used**: A detailed explanation of the methods and architectural changes applied at each step.  
# 3. **Thought Process**: A journal-style reflection describing your reasoning for applying each change.  
# 4. **Graphs**: Visualizations showing the changes in **response time** and **API cost** after each implementation.  
# 
# > ⚠️ **Important**: While the PDF should contain the graphs, the **code used to generate these graphs must be included in the notebook cells below this section**. This ensures that your results are reproducible.  
# 

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Timing results (in seconds)
data = {
    "Implementation": ["Base", "Base", "Base", "Cached", "Cached", "Cached"],
    "Run": [1, 2, 3, 1, 2, 3],
    "Time (s)": [
        18.790932178497314,
        19.09477663040161,
        21.34333610534668,
        18.501145839691162,
        17.842427492141724,
        16.417797803878784
    ]
}

df = pd.DataFrame(data)

# Compute averages
summary = df.groupby("Implementation")["Time (s)"].agg(["mean", "std"]).reset_index()
summary


# %%
sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(8,5))
sns.barplot(data=df, x="Implementation", y="Time (s)", ci="sd", palette="coolwarm")
plt.title("Average Response Time Comparison: Base vs Cached")
plt.ylabel("Response Time (seconds)")
plt.xlabel("Implementation Type")
plt.tight_layout()
plt.savefig("time_comparison_barplot.png")
plt.show()

# Line plot for all three runs
plt.figure(figsize=(8,5))
sns.lineplot(data=df, x="Run", y="Time (s)", hue="Implementation", marker="o")
plt.title("Response Time per Run")
plt.ylabel("Response Time (seconds)")
plt.xlabel("Run Number")
plt.tight_layout()
plt.savefig("time_per_run_lineplot.png")
plt.show()


# %% [markdown]
# ## End of Part 2
# 
# You must submit:  
# - The **current notebook file** (`.ipynb`).  
# - Its **Python conversion** (`.py` file).  
# - The **Report** (`.pdf`).
# - Run **files** (`.txt`).
# 
# All files should be placed inside a folder named "RollNumber_PA1". This folder must also include your **Part 1 files**, and the entire folder should be **zipped and submitted**.


