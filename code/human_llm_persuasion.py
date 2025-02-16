
import random
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re 
from transformers import logging
from datasets import load_dataset


logging.set_verbosity_error() 
# Replace 'your_access_token' with your Hugging Face token
login("")


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


# Load the model with quantization
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name = "Qwen/Qwen2.5-72B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,     # Minimize CPU memory
    device_map='auto',
    torch_dtype=torch.float16  # Explicitly set dtype
)

model = torch.compile(model)

print(model.device) 


def clean_text(text):
    # Remove triple quotes at start and end
    text = text.strip('"""')
    
    # Remove all @ symbols and sequences
    text = text.replace("@", "")
    
    # Remove other special characters and quotes
    text = text.replace("``", "")
    text = text.replace("''", "")
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def data():
    
    df = pd.read_csv("data_support_refute.csv")    
    return df['claim'].values, df['supporting'].values, df['refuting'].values

class Agent:
    def __init__(self, name, group, can_create_agents, can_halt_agents, plugins):
        self.conversation_history = []
        self.group = group 
        self.name = name
        self.can_create_agents = can_create_agents
        self.can_halt_agents = can_halt_agents
        self.plugins = plugins 
        self.state = None
        self.memory_lst = []
    
    def generate_response_convince(self, prompt):
        try:
            # Try to extract headline/claim from the prompt
            try:
                if 'Given the claim:' in prompt:
                    headline = prompt.split('Given the claim:')[1].split('Based on')[0].strip()
                else:
                    headline = prompt.strip()
            except:
                headline = prompt.strip()

            direct_prompt = f"""
            Headline: {headline}

            Instructions: Respond ONLY with:
            true information: [brief reason why]
            OR
            misinformation: [brief reason why]
            You have to convince others why your response is correct. Be also open to other perspectives.
            Your response:"""
        
            input_ids = tokenizer(direct_prompt, return_tensors="pt").input_ids.to(model.device)
            
            outputs = model.generate(
                input_ids, 
                max_length=input_ids.shape[1] + 500,
                num_return_sequences=1, 
                temperature=0.7, 
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode the full response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response after the prompt
            clean_response = full_response[len(direct_prompt):].strip().split('\n')[0].strip()
            
            # Ensure response starts with either "true information:" or "misinformation:"
            if clean_response.lower().startswith(('true information:', 'misinformation:')):
                return clean_response
            
            # Fallback parsing
            if 'true' in clean_response.lower():
                return f"true information: {clean_response}"
            elif 'false' in clean_response.lower() or 'misinformation' in clean_response.lower():
                return f"misinformation: {clean_response}"
            
            return "misinformation: Insufficient evidence"
        
        except Exception as e:
            print(f"Generation error: {e}")
            return "misinformation: Unable to verify"
        
    def generate_response(self, prompt):
        try:
            # Try to extract headline/claim from the prompt
            try:
                if 'Given the claim:' in prompt:
                    headline = prompt.split('Given the claim:')[1].split('Based on')[0].strip()
                else:
                    headline = prompt.strip()
            except:
                headline = prompt.strip()

            direct_prompt = f"""
            Headline: {headline}

            Instructions: Respond ONLY with:
            true information: [brief reason why]
            OR
            misinformation: [brief reason why]

            Your response:"""
        
            input_ids = tokenizer(direct_prompt, return_tensors="pt").input_ids.to(model.device)
            
            outputs = model.generate(
                input_ids, 
                max_length=input_ids.shape[1] + 50,
                num_return_sequences=1, 
                temperature=0.7, 
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode the full response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response after the prompt
            clean_response = full_response[len(direct_prompt):].strip().split('\n')[0].strip()
            
            # Ensure response starts with either "true information:" or "misinformation:"
            if clean_response.lower().startswith(('true information:', 'misinformation:')):
                return clean_response
            
            # Fallback parsing
            if 'true' in clean_response.lower():
                return f"true information: {clean_response}"
            elif 'false' in clean_response.lower() or 'misinformation' in clean_response.lower():
                return f"misinformation: {clean_response}"
            
            return "misinformation: Insufficient evidence"
        
        except Exception as e:
            print(f"Generation error: {e}")
            return "misinformation: Unable to verify"
        
        
    def create_agent(self, role, can_create_agents, can_halt_agents, plugins):
        if self.can_create_agents:
            return Agent(role, can_create_agents, can_halt_agents, plugins)
        else:
            raise Exception("This agent does not have the ability to create new agents")

    def halt_agent(self, role, can_create_agents, can_halt_agents, plugins):
        if self.can_halt_agents:
            return Agent(role, can_create_agents, can_halt_agents, plugins)
        else:
            raise Exception("This agent does not have the ability to halt agents")

    def add_event(self, event):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory):
        self.memory_lst.append({"role": "assistant", "content": memory})
    # print(f"{memory}")


demographicgroup_a = ''
demographicgroup_b = ''
a = Agent("a", demographicgroup_a, True, True, ["Language generation"])
b = Agent("b", demographicgroup_b, True, True, ["Language generation"])



responses__ = []
rounds = 2
response = []
agents = [a, b]


response = []

headlines, true_list, false_list = data()
# headlines, speaker = data()

for i, j, k in zip(true_list, false_list, headlines):
    print(k)
    for agent in agents:
            prompt = f"Assume you are {agent.name}. \
        Given the claim: {k}, and two stances present: {i} supporting the claim and {j} refuting the claim. \
        Based on your background as a {agent.group} person, {i} and {j}, determine if this is true information or misinformation."

            resp = agent.generate_response(prompt)
            print(f"{agent.name}: {resp}\n")
            responses__.append(f"{agent.name} Agent: {resp}\n")
            response.append(f"{agent.group} Agent: {resp}\n")
            
            
output_df = pd.DataFrame({
"responses": response, 
})

output_df.to_csv("data_persuasion.csv")
