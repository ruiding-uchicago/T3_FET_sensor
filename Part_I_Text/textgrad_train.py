#!/usr/bin/env python3
"""
Script to process scientific publication JSON files using Ollama.
Combines original text with human init prompt and generates structured output.
"""

import json
import os
import sys
import requests
import re
import random
import argparse
from pathlib import Path
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import re
from datetime import datetime

# OpenAI API Key - Set via environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Global API usage tracking
API_USAGE_TRACKER = {
    'ollama_calls': 0,
    'openai_calls': 0,
    'ollama_response_tokens': 0,
    'openai_request_tokens': 0,
    'openai_response_tokens': 0
}

# Summarizer prompts
SUMMARIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a given text (i.e. the variable). "
    "Your only responsibility is to critically aggregate and summarize the feedback from sources. "
    "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
    "You will be given a scenario, a prompt, the target language model type, the generated output, and the expert response. "
    "When giving a response, only provide the core summary of the feedback. Do not recommend a new version for the variable -- only summarize the feedback critically. "
)

SUMMARIZER_INSTRUCTION_TEMPLATE = (
    "# Task Overview\n"
    "Analyze what the current extraction prompt is lacking by comparing its outputs against human expert annotations across multiple scientific papers.\n\n"
    
    "## Research Topic\n"
    "**Focus Area**: {T}\n\n"
    
    "## Current Prompt Being Evaluated\n"
    "**Extraction Prompt**: {P}\n\n"
    
    "## Minibatch Analysis ({minibatch_size} papers)\n"
    "**Model Used**: {type_LLM}\n\n"
    
    "{minibatch_data}\n\n"
    
    "## Prompt Analysis Required\n"
    "Identify **exactly {issue_count}** specific, targeted issues in the current prompt by comparing AI outputs to expert annotations. Focus on the most critical problems that can be fixed with small modifications.\n\n"
    "For each of the {issue_count} issues, specify:\n"
    "1. **Exact prompt section** that needs modification (quote the specific text)\n"
    "2. **Specific problem** it causes in the outputs\n"
    "3. **Targeted fix** - what specific change would address this issue\n\n"
    "**IMPORTANT**: Only identify {issue_count} issues total. Prioritize the most impactful problems. Avoid broad criticisms - focus on specific, fixable elements that preserve working parts of the current prompt."
)

# Updater prompts
UPDATER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
)

UPDATER_INSTRUCTION_TEMPLATE = (
    "Here is the role of the variable you will improve: <ROLE>{role_description}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {P} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
    "<CONTEXT>{Instruction}</CONTEXT>\n\n"
    "**CRITICAL INSTRUCTION**: Make only **incremental, targeted modifications** to fix the specific issues mentioned in the feedback. Do NOT rewrite the entire prompt from scratch.\n\n"
    "**Learning Rate Constraint**: Modify at most {change_percentage}% of the original prompt. Focus on the smallest possible changes that address the feedback.\n\n"
    "**Modification Rules**:\n"
    "1. Keep the existing structure and working components intact\n"
    "2. Make only the minimal changes needed to address each specific issue\n"
    "3. Preserve any parts of the prompt that are already working well\n"
    "4. Focus on surgical edits rather than wholesale replacement\n"
    "5. **IMPORTANT**: Change no more than {change_percentage}% of the original text\n\n"
    "Send the improved variable in the following format:\n\n<IMPROVED_VARIABLE>{{the improved variable}}</IMPROVED_VARIABLE>\n\n"
    "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
)

def read_json_file(file_path):
    """Read and parse JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None

def read_human_prompt(prompt_file_path):
    """Read the human init prompt file."""
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading prompt file {prompt_file_path}: {e}")
        return None

def call_ollama(prompt, model="deepseek-r1:14b"):
    """Call Ollama API with the given prompt."""
    # Get OLLAMA_HOST from environment or use default
    ollama_host = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')
    url = f"http://{ollama_host}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '')
        
        # Track API usage
        API_USAGE_TRACKER['ollama_calls'] += 1
        if response_text:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            API_USAGE_TRACKER['ollama_response_tokens'] += len(response_text) // 4
        
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing Ollama response: {e}")
        return None

def sanitize_filename(doi):
    """Replace invalid filename characters in DOI."""
    return doi.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

def clean_response(response):
    """Remove content between <think> and </think> tags."""
    # Use regex to remove everything between <think> and </think> including the tags
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

def save_expert_data(records, doi, minibatch_name="minibatch", prompt_version="init_prompt"):
    """Save the expert records as markdown format."""
    # Create output directory structure
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create minibatch subdirectory
    minibatch_dir = output_dir / minibatch_name
    minibatch_dir.mkdir(exist_ok=True)
    
    # Create prompt version subdirectory
    version_dir = minibatch_dir / prompt_version
    version_dir.mkdir(exist_ok=True)
    
    # Create DOI-specific subdirectory
    doi_folder = version_dir / sanitize_filename(doi)
    doi_folder.mkdir(exist_ok=True)
    
    # Create filename and full path
    filepath = doi_folder / "JSON_expert.md"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Expert Annotations\n\n")
            f.write("```json\n")
            f.write(json.dumps({"records": records}, indent=2, ensure_ascii=False))
            f.write("\n```\n")
        print(f"Expert data saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"Error saving expert data to {filepath}: {e}")
        return None

def save_response(response, doi, minibatch_name="minibatch", prompt_version="init_prompt"):
    """Save the Ollama response to a markdown file in organized folder structure."""
    # Clean the response by removing <think> tags and content
    cleaned_response = clean_response(response)
    
    # Create output directory structure
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create minibatch subdirectory
    minibatch_dir = output_dir / minibatch_name
    minibatch_dir.mkdir(exist_ok=True)
    
    # Create prompt version subdirectory
    version_dir = minibatch_dir / prompt_version
    version_dir.mkdir(exist_ok=True)
    
    # Create DOI-specific subdirectory
    doi_folder = version_dir / sanitize_filename(doi)
    doi_folder.mkdir(exist_ok=True)
    
    # Create filename and full path
    filename = "JSON_output.md"
    filepath = doi_folder / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_response)
        print(f"Response saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"Error saving response to {filepath}: {e}")
        return None

def create_minibatch_data_section(minibatch_results):
    """Create the minibatch data section for summarizer prompt."""
    sections = []
    
    for i, result in enumerate(minibatch_results, 1):
        section = f"""### Paper {i}: DOI {result['doi']}

**Scientific Paper Content**: {result['original_text']}

**AI Generated Output**:
```json
{result['ai_output']}
```

**Expert Annotations**:
```json
{json.dumps(result['expert_output'], indent=2, ensure_ascii=False)}
```

---"""
        sections.append(section)
    
    return "\n\n".join(sections)

def get_issue_count_from_lr(learning_rate):
    """Convert learning rate to number of issues to address."""
    if learning_rate >= 0.8:
        return 3  # Aggressive: address 3 issues
    elif learning_rate >= 0.4:
        return 2  # Moderate: address 2 issues  
    else:
        return 1  # Conservative: address 1 issue

def get_change_percentage_from_lr(learning_rate):
    """Convert learning rate to percentage of prompt that can be changed."""
    # Scale learning rate to reasonable percentage bounds
    # LR 0.1 -> 5%, LR 0.5 -> 15%, LR 1.0 -> 25%
    percentage = int(5 + (learning_rate * 20))
    return min(percentage, 25)  # Cap at 25%

def create_minibatch_summarizer_prompt(minibatch_results, human_prompt, topic="FET Sensor", llm_type="deepseek-r1:14b", learning_rate=0.3, include_hardscore=False, minibatch_name=None, dois=None):
    """Create the summarizer prompt for minibatch processing."""
    minibatch_data = create_minibatch_data_section(minibatch_results)
    issue_count = get_issue_count_from_lr(learning_rate)
    
    # Add hard score section if requested
    hardscore_section = ""
    if include_hardscore and minibatch_name and dois:
        try:
            # Get hard scores for current prompt performance
            temp_results = evaluate_dueling_scores(
                minibatch_name, dois, ["init_prompt"],
                primary_metric="bertscore", 
                committee_metrics="committee_default"
            )
            
            if temp_results:
                hardscore_section = "\n\n## Current Prompt Performance (Hard Scores)\n"
                hardscore_section += "The current prompt achieves the following quantitative scores:\n"
                
                # Calculate averages
                metrics = ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']
                averages = {}
                
                for metric in metrics:
                    scores_list = [temp_results[doi]['init_prompt'][metric] for doi in temp_results if 'init_prompt' in temp_results[doi] and metric in temp_results[doi]['init_prompt']]
                    averages[metric] = sum(scores_list) / len(scores_list) if scores_list else 0.0
                
                hardscore_section += f"- BERTScore: {averages['bertscore']:.4f}\n"
                hardscore_section += f"- ROUGE-1: {averages['rouge_1']:.4f}\n" 
                hardscore_section += f"- METEOR: {averages['meteor']:.4f}\n"
                hardscore_section += f"- Committee Score: {averages['committee']:.4f}\n"
                hardscore_section += "\nConsider these baseline performance metrics when identifying improvement areas.\n"
        except Exception as e:
            print(f"Warning: Could not include hard scores in summarizer: {e}")
    
    full_template = SUMMARIZER_INSTRUCTION_TEMPLATE + hardscore_section
    
    return full_template.format(
        T=topic,
        P=human_prompt,
        type_LLM=llm_type,
        minibatch_size=len(minibatch_results),
        minibatch_data=minibatch_data,
        issue_count=issue_count
    )

def save_summarizer_prompt(prompt, minibatch_name="minibatch", iteration=1):
    """Save the summarizer prompt for review."""
    # Create output directory structure
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create minibatch subdirectory
    minibatch_dir = output_dir / minibatch_name
    minibatch_dir.mkdir(exist_ok=True)
    
    # Create iteration subdirectory
    iter_dir = minibatch_dir / f"iter{iteration}"
    iter_dir.mkdir(exist_ok=True)
    
    # Create filename and full path
    filepath = iter_dir / "Prompt_Summarizer.md"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Summarizer prompt saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"Error saving summarizer prompt to {filepath}: {e}")
        return None

def call_openai_summarizer(prompt, model_name="o4-mini", temperature=1.0):
    """Call OpenAI API for summarizer analysis."""
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("Please set your OpenAI API key in the script before using the summarizer.")
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Track API usage
        API_USAGE_TRACKER['openai_calls'] += 1
        if hasattr(response, 'usage') and response.usage:
            API_USAGE_TRACKER['openai_request_tokens'] += response.usage.prompt_tokens
            API_USAGE_TRACKER['openai_response_tokens'] += response.usage.completion_tokens
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def call_ollama_summarizer(prompt, model="deepseek-r1:14b", temperature=1.0):
    """Call Ollama API for summarizer analysis."""
    # Combine system prompt and user prompt for Ollama
    combined_prompt = f"{SUMMARIZER_SYSTEM_PROMPT}\n\n{prompt}"
    
    # Get OLLAMA_HOST from environment or use default
    ollama_host = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')
    url = f"http://{ollama_host}/api/generate"
    
    payload = {
        "model": model,
        "prompt": combined_prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '')
        
        # Track API usage
        API_USAGE_TRACKER['ollama_calls'] += 1
        if response_text:
            API_USAGE_TRACKER['ollama_response_tokens'] += len(response_text) // 4
        
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API for summarizer: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing Ollama response for summarizer: {e}")
        return None

def save_summarizer_response(response, minibatch_name="minibatch", iteration=1):
    """Save the summarizer response."""
    # Create output directory structure
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create minibatch subdirectory
    minibatch_dir = output_dir / minibatch_name
    minibatch_dir.mkdir(exist_ok=True)
    
    # Create iteration subdirectory
    iter_dir = minibatch_dir / f"iter{iteration}"
    iter_dir.mkdir(exist_ok=True)
    
    # Create filename and full path
    filepath = iter_dir / "Summarizer_Analysis.md"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"Summarizer analysis saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"Error saving summarizer analysis to {filepath}: {e}")
        return None

def create_updater_prompt(current_prompt, feedback, topic="FET Sensor", learning_rate=0.3):
    """Create the updater prompt using the template."""
    role_description = f"improve template for extraction from scientific publication in {topic} domain"
    change_percentage = get_change_percentage_from_lr(learning_rate)
    
    return UPDATER_INSTRUCTION_TEMPLATE.format(
        role_description=role_description,
        P=current_prompt,
        Instruction=feedback,
        change_percentage=change_percentage
    )

def save_updater_prompt(prompt, minibatch_name="minibatch", iteration=1):
    """Save the updater prompt for review."""
    # Create output directory structure
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create minibatch subdirectory
    minibatch_dir = output_dir / minibatch_name
    minibatch_dir.mkdir(exist_ok=True)
    
    # Create iteration subdirectory
    iter_dir = minibatch_dir / f"iter{iteration}"
    iter_dir.mkdir(exist_ok=True)
    
    # Create filename and full path
    filepath = iter_dir / "Prompt_Updater.md"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Updater prompt saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"Error saving updater prompt to {filepath}: {e}")
        return None

def call_openai_updater(prompt, model_name="o4-mini", temperature=1.0):
    """Call OpenAI API for updater optimization."""
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("Please set your OpenAI API key in the script before using the updater.")
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": UPDATER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Track API usage
        API_USAGE_TRACKER['openai_calls'] += 1
        if hasattr(response, 'usage') and response.usage:
            API_USAGE_TRACKER['openai_request_tokens'] += response.usage.prompt_tokens
            API_USAGE_TRACKER['openai_response_tokens'] += response.usage.completion_tokens
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API for updater: {e}")
        return None

def call_ollama_updater(prompt, model="deepseek-r1:14b", temperature=1.0):
    """Call Ollama API for updater optimization."""
    # Combine system prompt and user prompt for Ollama
    combined_prompt = f"{UPDATER_SYSTEM_PROMPT}\n\n{prompt}"
    
    # Get OLLAMA_HOST from environment or use default
    ollama_host = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')
    url = f"http://{ollama_host}/api/generate"
    
    payload = {
        "model": model,
        "prompt": combined_prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '')
        
        # Track API usage
        API_USAGE_TRACKER['ollama_calls'] += 1
        if response_text:
            API_USAGE_TRACKER['ollama_response_tokens'] += len(response_text) // 4
        
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API for updater: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing Ollama response for updater: {e}")
        return None

def extract_improved_variable(response):
    """Extract the improved variable from between <IMPROVED_VARIABLE> tags."""
    import re
    match = re.search(r'<IMPROVED_VARIABLE>(.*?)</IMPROVED_VARIABLE>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        print("Warning: Could not find <IMPROVED_VARIABLE> tags in response")
        return response  # Return full response as fallback

def save_optimized_prompt(improved_prompt, minibatch_name="minibatch", iteration=1):
    """Save the optimized prompt to the minibatch folder."""
    # Create optimized_prompt directory structure
    optimized_dir = Path("optimized_prompt")
    optimized_dir.mkdir(exist_ok=True)
    
    # Create minibatch subdirectory
    minibatch_dir = optimized_dir / minibatch_name
    minibatch_dir.mkdir(exist_ok=True)
    
    # Create filename and full path
    filepath = minibatch_dir / f"Prompt_{minibatch_name}_iter{iteration}.md"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(improved_prompt)
        print(f"Optimized prompt saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"Error saving optimized prompt to {filepath}: {e}")
        return None

def get_training_files(fold_path="split_datasets/fold_3/train"):
    """Get list of JSON files from training set."""
    try:
        train_dir = Path(fold_path)
        if not train_dir.exists():
            print(f"Training directory not found: {fold_path}")
            return []
        
        json_files = list(train_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in training set")
        return json_files
    except Exception as e:
        print(f"Error accessing training files: {e}")
        return []

def sample_minibatch(training_files, batch_size):
    """Randomly sample files for minibatch processing."""
    if len(training_files) < batch_size:
        print(f"Warning: Only {len(training_files)} files available, using all of them")
        return training_files
    
    sampled = random.sample(training_files, batch_size)
    print(f"Sampled {len(sampled)} files for minibatch processing")
    return sampled

def calculate_bleu_score(reference_text, candidate_text):
    """Calculate BLEU score between reference and candidate text."""
    try:
        # Check if texts are valid and non-empty
        if not reference_text or not candidate_text or not reference_text.strip() or not candidate_text.strip():
            return 0.0
        
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        # Tokenize texts
        reference_tokens = nltk.word_tokenize(reference_text.lower())
        candidate_tokens = nltk.word_tokenize(candidate_text.lower())
        
        # Check if tokenization produced valid tokens
        if not reference_tokens or not candidate_tokens:
            return 0.0
        
        # Calculate BLEU score with smoothing
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)
        
        return bleu_score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_rouge_score(reference_text, candidate_text):
    """Calculate ROUGE-1 score between reference and candidate text."""
    try:
        # Check if texts are valid and non-empty
        if not reference_text or not candidate_text or not reference_text.strip() or not candidate_text.strip():
            return 0.0
        
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(reference_text, candidate_text)
        rouge_1_f1 = scores['rouge1'].fmeasure
        return rouge_1_f1
    except Exception as e:
        print(f"Error calculating ROUGE score: {e}")
        return 0.0

def calculate_meteor_score(reference_text, candidate_text):
    """Calculate METEOR score."""
    try:
        from nltk.translate.meteor_score import meteor_score
        
        # Download required data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        if not reference_text or not candidate_text or not reference_text.strip() or not candidate_text.strip():
            return 0.0
        
        # Tokenize
        reference_tokens = nltk.word_tokenize(reference_text.lower())
        candidate_tokens = nltk.word_tokenize(candidate_text.lower())
        
        if not reference_tokens or not candidate_tokens:
            return 0.0
            
        score = meteor_score([reference_tokens], candidate_tokens)
        return score
        
    except Exception as e:
        print(f"Error calculating METEOR: {e}")
        return 0.0

def calculate_bertscore(reference_text, candidate_text):
    """Calculate BERTScore F1."""
    try:
        from bert_score import score
        
        if not reference_text or not candidate_text or not reference_text.strip() or not candidate_text.strip():
            return 0.0
        
        # Use a small, fast model
        P, R, F1 = score([candidate_text], [reference_text], 
                         model_type="distilbert-base-uncased", 
                         verbose=False)
        
        return F1.item()
        
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return 0.0

def extract_json_from_content(content):
    """Extract JSON content from markdown or raw text."""
    if not content or not content.strip():
        return None
    
    # Try to find JSON in markdown format first
    json_start = content.find('```json\n')
    if json_start != -1:
        json_start += 8  # Move past ```json\n
        json_end = content.find('\n```', json_start)
        if json_end != -1:
            return content[json_start:json_end].strip()
    
    # Try to find JSON block without markdown
    content = content.strip()
    if content.startswith('{') and content.endswith('}'):
        return content
    
    # Look for JSON patterns in the content
    json_pattern = r'\{.*?\}'
    matches = re.findall(json_pattern, content, re.DOTALL)
    if matches:
        # Return the longest match (likely the most complete JSON)
        return max(matches, key=len)
    
    return None

def calculate_exact_field_match(expert_json_str, ai_json_str):
    """Calculate exact field match percentage for JSON structures."""
    try:
        # Extract and parse expert JSON
        expert_json_clean = extract_json_from_content(expert_json_str)
        if not expert_json_clean:
            return 0.0
        expert_data = json.loads(expert_json_clean)
        
        # Extract and parse AI JSON
        ai_json_clean = extract_json_from_content(ai_json_str)
        if not ai_json_clean:
            return 0.0
        ai_data = json.loads(ai_json_clean)
        
        # Extract records arrays
        expert_records = expert_data.get('records', [])
        ai_records = ai_data.get('records', [])
        
        if not expert_records or not ai_records:
            return 0.0
        
        # Compare first record (most common case)
        expert_record = expert_records[0] if expert_records else {}
        ai_record = ai_records[0] if ai_records else {}
        
        total_fields = len(expert_record)
        if total_fields == 0:
            return 0.0
        
        exact_matches = 0
        for field, expert_value in expert_record.items():
            ai_value = ai_record.get(field, None)
            if ai_value is not None:
                # Normalize values for comparison
                expert_norm = str(expert_value).strip().lower()
                ai_norm = str(ai_value).strip().lower()
                if expert_norm == ai_norm:
                    exact_matches += 1
        
        return exact_matches / total_fields
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in exact field match: {e}")
        return 0.0
    except Exception as e:
        print(f"Error calculating exact field match: {e}")
        return 0.0

def calculate_jaccard_similarity(reference_text, candidate_text):
    """Calculate Jaccard similarity based on word sets."""
    try:
        if not reference_text or not candidate_text or not reference_text.strip() or not candidate_text.strip():
            return 0.0
        
        # Tokenize and create sets
        ref_tokens = set(nltk.word_tokenize(reference_text.lower()))
        cand_tokens = set(nltk.word_tokenize(candidate_text.lower()))
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Calculate Jaccard = intersection / union
        intersection = len(ref_tokens.intersection(cand_tokens))
        union = len(ref_tokens.union(cand_tokens))
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        print(f"Error calculating Jaccard: {e}")
        return 0.0

def calculate_all_scores(expert_json_str, ai_json_str):
    """Calculate all available scoring metrics."""
    scores = {}
    
    # Basic scores
    scores['bleu'] = calculate_bleu_score(expert_json_str, ai_json_str)
    scores['rouge_1'] = calculate_rouge_score(expert_json_str, ai_json_str)
    
    # Enhanced scores
    scores['meteor'] = calculate_meteor_score(expert_json_str, ai_json_str)
    scores['bertscore'] = calculate_bertscore(expert_json_str, ai_json_str)
    scores['exact_match'] = calculate_exact_field_match(expert_json_str, ai_json_str)
    scores['jaccard'] = calculate_jaccard_similarity(expert_json_str, ai_json_str)
    
    return scores

def calculate_committee_score(scores, metrics_list=None):
    """Calculate committee score by averaging specified metrics."""
    if metrics_list is None:
        # Default committee: bertscore, rouge_1, meteor (most trusted)
        metrics_list = ['bertscore', 'rouge_1', 'meteor']
    
    # Handle special committee names
    if isinstance(metrics_list, str):
        if metrics_list == "committee_default":
            metrics_list = ['bertscore', 'rouge_1', 'meteor']
        elif metrics_list == "all":
            metrics_list = ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard']
        else:
            # Single metric
            metrics_list = [metrics_list]
    
    valid_scores = []
    for metric in metrics_list:
        if metric in scores and scores[metric] is not None:
            valid_scores.append(scores[metric])
    
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

def evaluate_dueling_scores(minibatch_name, dois, prompt_versions=["init_prompt", "iter1", "iter2"], 
                           primary_metric="bertscore", committee_metrics="committee_default"):
    """Calculate all scoring metrics for different prompt versions."""
    print(f"\n{'='*50}")
    print("DUELING: EVALUATING PROMPT PERFORMANCE")
    print(f"Primary Metric: {primary_metric.upper()} | Committee: {committee_metrics}")
    print(f"{'='*50}")
    
    results = {}
    
    for doi in dois:
        sanitized_doi = sanitize_filename(doi)
        results[doi] = {}
        
        print(f"\nEvaluating DOI: {doi}")
        
        # Get expert JSON (ground truth)
        expert_path = Path(f"output/{minibatch_name}/init_prompt/{sanitized_doi}/JSON_expert.md")
        if not expert_path.exists():
            print(f"Expert data not found for {doi}")
            continue
        
        try:
            with open(expert_path, 'r', encoding='utf-8') as f:
                expert_content = f.read()
            expert_json_str = extract_json_from_content(expert_content)
            if not expert_json_str:
                print(f"Could not extract expert JSON for {doi}")
                continue
        except Exception as e:
            print(f"Error reading expert data for {doi}: {e}")
            continue
        
        # Evaluate each prompt version
        for version in prompt_versions:
            ai_output_path = Path(f"output/{minibatch_name}/{version}/{sanitized_doi}/JSON_output.md")
            
            if ai_output_path.exists():
                try:
                    with open(ai_output_path, 'r', encoding='utf-8') as f:
                        ai_output = f.read().strip()
                    
                    ai_json_str = extract_json_from_content(ai_output)
                    if not ai_json_str:
                        print(f"  {version:12} - Could not extract AI JSON")
                        continue
                    
                    # Calculate all scores
                    all_scores = calculate_all_scores(expert_json_str, ai_json_str)
                    
                    # Calculate committee score
                    committee_score = calculate_committee_score(all_scores, committee_metrics)
                    all_scores['committee'] = committee_score
                    
                    results[doi][version] = all_scores
                    
                    # Display results
                    primary_score = all_scores.get(primary_metric, 0.0)
                    print(f"  {version:12} - {primary_metric.upper()}: {primary_score:.4f}, Committee: {committee_score:.4f}")
                    print(f"                 BLEU: {all_scores['bleu']:.3f} | ROUGE: {all_scores['rouge_1']:.3f} | METEOR: {all_scores['meteor']:.3f}")
                    print(f"                 BERTScore: {all_scores['bertscore']:.3f} | ExactMatch: {all_scores['exact_match']:.3f} | Jaccard: {all_scores['jaccard']:.3f}")
                    
                except Exception as e:
                    print(f"  {version:12} - Error: {e}")
                    # Initialize with zeros for all metrics
                    zero_scores = {metric: 0.0 for metric in ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']}
                    results[doi][version] = zero_scores
            else:
                print(f"  {version:12} - Output not found")
                zero_scores = {metric: 0.0 for metric in ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']}
                results[doi][version] = zero_scores
    
    # Calculate averages
    print(f"\n{'='*60}")
    print("AVERAGE SCORES ACROSS ALL PAPERS:")
    print(f"{'='*60}")
    
    version_averages = {}
    for version in prompt_versions:
        if not any(version in results[doi] for doi in results):
            continue
            
        # Calculate averages for each metric
        metrics = ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']
        averages = {}
        
        for metric in metrics:
            scores_list = [results[doi][version][metric] for doi in results if version in results[doi] and metric in results[doi][version]]
            averages[metric] = sum(scores_list) / len(scores_list) if scores_list else 0.0
        
        version_averages[version] = averages
        
        print(f"\n🔸 {version.upper()}:")
        print(f"  Primary ({primary_metric.upper()}): {averages.get(primary_metric, 0.0):.4f} | Committee: {averages['committee']:.4f}")
        print(f"  BLEU: {averages['bleu']:.3f} | ROUGE-1: {averages['rouge_1']:.3f} | METEOR: {averages['meteor']:.3f}")
        print(f"  BERTScore: {averages['bertscore']:.3f} | ExactMatch: {averages['exact_match']:.3f} | Jaccard: {averages['jaccard']:.3f}")
    
    # Determine winners
    if version_averages:
        primary_winner = max(version_averages.keys(), key=lambda v: version_averages[v].get(primary_metric, 0.0))
        committee_winner = max(version_averages.keys(), key=lambda v: version_averages[v]['committee'])
        
        print(f"\n🎯 PRIMARY METRIC WINNER ({primary_metric.upper()}): {primary_winner.upper()} ({version_averages[primary_winner].get(primary_metric, 0.0):.4f})")
        print(f"🏆 COMMITTEE WINNER: {committee_winner.upper()} ({version_averages[committee_winner]['committee']:.4f})")
    
    # Save results
    save_dueling_results(results, minibatch_name)
    
    return results

def save_dueling_results(results, minibatch_name):
    """Save dueling results to file."""
    output_dir = Path("output") / minibatch_name
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / "hard_score.json"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📊 Hard score results saved to: {filepath}")
    except Exception as e:
        print(f"Error saving dueling results: {e}")

def process_single_file(json_file_path, prompt_file_path, minibatch_name="minibatch", prompt_version="init_prompt", model="deepseek-r1:14b"):
    """Process a single JSON file for minibatch processing."""
    print(f"Processing: {json_file_path} with {prompt_version}")
    log_to_cluster(f"Processing {json_file_path.name} with {prompt_version} using model {model}")
    
    # Read JSON file
    data = read_json_file(json_file_path)
    if not data:
        return None
    
    # Extract required fields
    doi = data.get('DOI', 'unknown_doi')
    original_text = data.get('Original_Text', '')
    records = data.get('records', [])
    
    if not original_text:
        print(f"No 'Original_Text' found in {json_file_path}")
        return None
    
    # Save expert data for each version to facilitate checking
    if records:
        expert_file = save_expert_data(records, doi, minibatch_name, prompt_version)
        if expert_file:
            print(f"Expert data saved: {expert_file}")
    else:
        print(f"No 'records' found in {json_file_path}")
    
    # Read human prompt
    human_prompt = read_human_prompt(prompt_file_path)
    if not human_prompt:
        return None
    
    # Combine original text and human prompt
    combined_prompt = f"{original_text}\n\n{human_prompt}"
    
    print(f"Sending request to Ollama for DOI: {doi}")
    
    # Call Ollama with specified model
    response = call_ollama(combined_prompt, model)
    if not response:
        return None
    
    # Save response
    output_file = save_response(response, doi, minibatch_name, prompt_version)
    if not output_file:
        return None
    
    # Read the saved AI output for summarizer
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            ai_output_content = f.read()
    except Exception as e:
        print(f"Error reading AI output file: {e}")
        ai_output_content = response  # Fallback to raw response
    
    print(f"Successfully processed {json_file_path} -> {output_file}")
    log_to_cluster(f"✅ Successfully processed {json_file_path.name} -> DOI: {doi}")
    
    # Return data for minibatch processing
    return {
        'doi': doi,
        'original_text': original_text,
        'ai_output': ai_output_content,
        'expert_output': {"records": records} if records else {},
        'human_prompt': human_prompt
    }

def process_minibatch(file_paths, prompt_file_path="human_init_prompt.md", minibatch_name="minibatch", temperature=1.0, iterations=2, primary_metric="bertscore", committee_metrics="committee_default", llm_judger=False, judger_duel_set="minibatch", judger_rounds=1, judger_show_templates=False, judger_template1="iter1", judger_template2="iter2", dev_path="split_datasets/fold_3/dev", child_parent_duel=False, learning_rate=0.3, duel_mode="no_duel", winner_basis="hard_score", enforce_child_replacement=False, include_hardscore_in_summarizer=False, model="deepseek-r1:14b", summarizer_api="openai", updater_api="openai", meta_model="deepseek-r1:14b"):
    """Process multiple files in a minibatch."""
    print(f"\n{'='*50}")
    print(f"PROCESSING MINIBATCH: {minibatch_name}")
    print(f"Files: {len(file_paths)}")
    print(f"Model: {model}")
    print(f"{'='*50}")
    
    log_to_cluster(f"🚀 Starting minibatch {minibatch_name} with {len(file_paths)} files using model {model}")
    
    minibatch_results = []
    human_prompt = None
    dois = []  # Track DOIs for dueling evaluation
    
    # Process each file in the minibatch with initial prompt
    for file_path in file_paths:
        result = process_single_file(file_path, prompt_file_path, minibatch_name, "init_prompt", model)
        if result:
            minibatch_results.append(result)
            dois.append(result['doi'])
            if not human_prompt:  # Get human prompt from first successful processing
                human_prompt = result['human_prompt']
    
    if not minibatch_results:
        print("No files were successfully processed in this minibatch")
        log_to_cluster("❌ No files successfully processed in minibatch", "ERROR")
        return False
    
    print(f"\nProcessed {len(minibatch_results)}/{len(file_paths)} files successfully")
    log_to_cluster(f"📊 Processed {len(minibatch_results)}/{len(file_paths)} files successfully")
    
    # Now run Summarizer-Updater iterations for the entire minibatch
    print(f"\n{'='*50}")
    print(f"RUNNING {iterations} ITERATIONS OF OPTIMIZATION")
    print(f"{'='*50}")
    
    log_to_cluster(f"🔄 Starting {iterations} optimization iterations")
    
    current_prompt = human_prompt
    
    for iteration in range(1, iterations + 1):
        print(f"\n{'='*30}")
        print(f"ITERATION {iteration}/{iterations}")
        print(f"{'='*30}")
        
        log_to_cluster(f"⚙️ Starting optimization iteration {iteration}/{iterations}")
        
        # Create minibatch summarizer prompt
        summarizer_prompt = create_minibatch_summarizer_prompt(
            minibatch_results, current_prompt, learning_rate=learning_rate,
            include_hardscore=include_hardscore_in_summarizer,
            minibatch_name=minibatch_name, dois=dois
        )
        
        # Save summarizer prompt for review
        prompt_file = save_summarizer_prompt(summarizer_prompt, minibatch_name, iteration)
        if prompt_file:
            print(f"Summarizer prompt saved: {prompt_file}")
            
            # Call API for analysis
            if summarizer_api == "ollama":
                print(f"Calling Ollama API for summarizer analysis (iteration {iteration})...")
                summarizer_response = call_ollama_summarizer(summarizer_prompt, meta_model, temperature)
            else:
                print(f"Calling OpenAI API for summarizer analysis (iteration {iteration})...")
                summarizer_response = call_openai_summarizer(summarizer_prompt, temperature=temperature)
            
            if summarizer_response:
                analysis_file = save_summarizer_response(summarizer_response, minibatch_name, iteration)
                if analysis_file:
                    print(f"Summarizer analysis saved: {analysis_file}")
                else:
                    print("Failed to save summarizer analysis")
                
                # Now run the Updater component
                print(f"Running Updater to optimize the prompt (iteration {iteration})...")
                
                # Create updater prompt
                updater_prompt = create_updater_prompt(
                    current_prompt=current_prompt,
                    feedback=summarizer_response,
                    learning_rate=learning_rate
                )
                
                # Save updater prompt for review
                updater_prompt_file = save_updater_prompt(updater_prompt, minibatch_name, iteration)
                if updater_prompt_file:
                    print(f"Updater prompt saved: {updater_prompt_file}")
                    
                    # Call API for optimization
                    if updater_api == "ollama":
                        print(f"Calling Ollama API for prompt optimization (iteration {iteration})...")
                        updater_response = call_ollama_updater(updater_prompt, meta_model, temperature)
                    else:
                        print(f"Calling OpenAI API for prompt optimization (iteration {iteration})...")
                        updater_response = call_openai_updater(updater_prompt, temperature=temperature)
                    
                    if updater_response:
                        # Extract the improved variable
                        improved_prompt = extract_improved_variable(updater_response)
                        
                        # Save optimized prompt
                        optimized_file = save_optimized_prompt(improved_prompt, minibatch_name, iteration)
                        if optimized_file:
                            print(f"Optimized prompt saved: {optimized_file}")
                            log_to_cluster(f"✅ Iteration {iteration} completed - optimized prompt saved")
                            
                            # Update current prompt for next iteration
                            current_prompt = improved_prompt
                            print(f"✅ Iteration {iteration} completed successfully!")
                        else:
                            print(f"Failed to save optimized prompt for iteration {iteration}")
                            return False
                    else:
                        print(f"Updater optimization failed for iteration {iteration} - check API key or connection")
                        return False
                else:
                    print(f"Failed to save updater prompt for iteration {iteration}")
                    return False
            else:
                print(f"Summarizer analysis failed for iteration {iteration} - check API key or connection")
                return False
        else:
            print(f"Failed to save summarizer prompt for iteration {iteration}")
            return False
    
    print(f"\n🎉 ALL {iterations} ITERATIONS COMPLETED SUCCESSFULLY!")
    print(f"📝 Final optimized prompt: ./optimized_prompt/{minibatch_name}/Prompt_{minibatch_name}_iter{iterations}.md")
    
    log_to_cluster(f"🎉 All {iterations} optimization iterations completed successfully")
    
    # Now re-evaluate with optimized prompts
    print(f"\n{'='*50}")
    print("RE-EVALUATING WITH OPTIMIZED PROMPTS")
    print(f"{'='*50}")
    
    log_to_cluster(f"🔄 Starting re-evaluation with optimized prompts")
    
    for iteration in range(1, iterations + 1):
        optimized_prompt_path = Path(f"optimized_prompt/{minibatch_name}/Prompt_{minibatch_name}_iter{iteration}.md")
        
        if optimized_prompt_path.exists():
            print(f"\n{'='*30}")
            print(f"RE-EVALUATING WITH ITER{iteration} OPTIMIZED PROMPT")
            print(f"{'='*30}")
            
            # Process each file with the optimized prompt
            for file_path in file_paths:
                result = process_single_file(file_path, str(optimized_prompt_path), minibatch_name, f"iter{iteration}", model)
                if result:
                    print(f"✅ Re-evaluated {file_path.name} with iter{iteration} prompt")
                else:
                    print(f"❌ Failed to re-evaluate {file_path.name} with iter{iteration} prompt")
        else:
            print(f"Warning: Optimized prompt for iter{iteration} not found at {optimized_prompt_path}")
    
    # Parse committee metrics
    if committee_metrics and ',' in committee_metrics:
        parsed_committee_metrics = [m.strip() for m in committee_metrics.split(',')]
    else:
        parsed_committee_metrics = committee_metrics
    
    # Determine versions to evaluate based on iterations
    if iterations == 1:
        # TextGrad baseline: only compare iter1 vs init_prompt
        versions_to_evaluate = ["init_prompt", "iter1"]
        print(f"\n📊 TEXTGRAD BASELINE MODE: Comparing ITER1 vs INIT_PROMPT only")
    else:
        # Standard mode: compare all three versions
        versions_to_evaluate = ["init_prompt", "iter1", "iter2"]
    
    # Run dueling evaluation
    dueling_results = evaluate_dueling_scores(
        minibatch_name, dois, versions_to_evaluate,
        primary_metric=primary_metric, 
        committee_metrics=parsed_committee_metrics
    )
    
    log_to_cluster(f"📊 Dueling evaluation completed for {len(versions_to_evaluate)} prompt versions")
    
    # Initialize judger results
    llm_results = None
    hierarchical_results = None
    
    # Run LLM Judger evaluation based on duel mode
    if duel_mode != "no_duel" and iterations > 1:
        print(f"\n{'='*50}")
        print("🧠 STARTING LLM JUDGER EVALUATION")
        print(f"{'='*50}")
        log_to_cluster(f"🧠 Starting LLM judger evaluation in {duel_mode} mode")
    elif duel_mode != "no_duel" and iterations == 1:
        print(f"\n⚠️  WARNING: LLM dueling disabled - requires iterations > 1 (TextGrad baseline mode)")
        print(f"    Using hard score comparison between iter1 and init_prompt")
    
    # Actually run LLM judger if duel mode is enabled
    if duel_mode != "no_duel" and iterations > 1:
        
        # Determine DOI sets for judging 
        effective_judger_set = judger_duel_set
        judger_dois_sets = {}
        
        if effective_judger_set in ["minibatch", "both"]:
            judger_dois_sets["minibatch"] = dois
        
        if effective_judger_set in ["dev", "both"]:
            # Get dev set DOIs and ensure they have been processed
            dev_files = get_dev_files(dev_path)
            dev_dois = get_dois_from_files(dev_files)
            
            if dev_dois:
                # Process dev files with both templates if not already done
                print(f"\n📊 Processing dev set with optimized templates...")
                for iteration in range(1, iterations + 1):
                    optimized_prompt_path = Path(f"optimized_prompt/{minibatch_name}/Prompt_{minibatch_name}_iter{iteration}.md")
                    
                    if optimized_prompt_path.exists():
                        print(f"\n📄 Processing dev set with iter{iteration} template...")
                        for dev_file in dev_files:
                            result = process_single_file(dev_file, str(optimized_prompt_path), minibatch_name, f"iter{iteration}", model)
                            if result:
                                print(f"  ✅ Processed {dev_file.name}")
                            else:
                                print(f"  ❌ Failed to process {dev_file.name}")
                
                judger_dois_sets["dev"] = dev_dois
            else:
                print("⚠️  No DOIs found in dev set, skipping dev evaluation")
        
        # Run LLM judger for each specified set
        for set_name, judger_dois in judger_dois_sets.items():
            if judger_dois:
                print(f"\n{'='*40}")
                print(f"🥊 LLM JUDGER: {set_name.upper()} SET")
                print(f"{'='*40}")
                
                if duel_mode == "child_parent":
                    # Run hierarchical tournament: child vs child, then winner vs parent
                    hierarchical_results = run_hierarchical_llm_duel(
                        minibatch_name=minibatch_name,
                        dois=judger_dois,
                        child1=judger_template1,
                        child2=judger_template2,
                        parent="init_prompt",
                        show_templates=judger_show_templates,
                        duel_rounds=judger_rounds,
                        duel_set=set_name
                    )
                    
                    if hierarchical_results:
                        print(f"✅ Hierarchical tournament completed for {set_name} set")
                        print(f"🏆 FINAL CHAMPION: {hierarchical_results['final_champion'].upper()}")
                        log_to_cluster(f"🏆 Hierarchical tournament completed - champion: {hierarchical_results['final_champion'].upper()}")
                    else:
                        print(f"❌ Hierarchical tournament failed for {set_name} set")
                elif duel_mode == "child_only":
                    # Run simple child generation duel only
                    llm_results = run_llm_judger_duel(
                        minibatch_name=minibatch_name,
                        dois=judger_dois,
                        template1_version=judger_template1,
                        template2_version=judger_template2,
                        show_templates=judger_show_templates,
                        duel_rounds=judger_rounds,
                        duel_set=set_name
                    )
                    
                    if llm_results:
                        print(f"✅ LLM judger completed for {set_name} set")
                        log_to_cluster(f"✅ LLM judger completed for {set_name} set")
                    else:
                        print(f"❌ LLM judger failed for {set_name} set")
        
        print(f"\n🧠 LLM JUDGER EVALUATION COMPLETED!")
    
    # Determine minibatch winner
    final_winner = determine_minibatch_winner(
        minibatch_name=minibatch_name,
        dois=dois,
        duel_mode=duel_mode,
        winner_basis=winner_basis,
        enforce_child_replacement=enforce_child_replacement,
        primary_metric=primary_metric,
        committee_metrics=parsed_committee_metrics,
        llm_results=llm_results,
        hierarchical_results=hierarchical_results
    )
    
    print(f"\n🎉 MINIBATCH {minibatch_name} FULLY COMPLETED!")
    print(f"🏆 WINNER: {final_winner.upper()}")
    
    log_to_cluster(f"🎉 Minibatch {minibatch_name} completed successfully - Winner: {final_winner.upper()}")
    
    return final_winner

# LLM Judger System - Prompt Templates
JUDGER_SYSTEM_PROMPT = (
    "You are an expert evaluator specializing in scientific literature data extraction. "
    "Your task is to compare two different extraction templates by analyzing their outputs against expert annotations. "
    "You must provide a clear judgment on which template performs better for the given scientific paper."
)

JUDGER_INSTRUCTION_TEMPLATE = (
    "# Expert Evaluation Task\n\n"
    "**Publication DOI**: {doi}\n\n"
    "**Full Scientific Paper Text**:\n"
    "{original_text}\n\n"
    "## Template Comparison\n\n"
    "**Template #1 Output**:\n"
    "```json\n"
    "{template1_output}\n"
    "```\n\n"
    "**Template #2 Output**:\n"
    "```json\n"
    "{template2_output}\n"
    "```\n\n"
    "**Expert Ground Truth**:\n"
    "```json\n"
    "{expert_annotations}\n"
    "```\n\n"
    "{optional_template_content}"
    "## Evaluation Task\n\n"
    "Please carefully analyze both template outputs against the expert annotations. Consider:\n"
    "1. **Accuracy**: Which template extracts information more accurately?\n"
    "2. **Completeness**: Which template captures more relevant information?\n"
    "3. **Domain Knowledge**: Which template better understands chemistry/materials science concepts?\n"
    "4. **Structure**: Which template follows the expected JSON format better?\n\n"
    "**REQUIRED OUTPUT FORMAT**:\n"
    "```json\n"
    "{{\n"
    "  \"winner\": \"#1\" or \"#2\",\n"
    "  \"reasoning\": \"Your detailed explanation (2-3 sentences) comparing both templates and why the winner is better\"\n"
    "}}\n"
    "```\n\n"
    "You MUST choose a winner. Ties are not allowed."
)

def create_llm_judger_prompt(doi, original_text, template1_output, template2_output, 
                            expert_annotations, show_templates=False, template1_content="", template2_content=""):
    """Create the LLM judger prompt for comparing two templates."""
    
    optional_template_content = ""
    if show_templates:
        optional_template_content = (
            f"**Template #1 Content**:\n"
            f"```\n{template1_content}\n```\n\n"
            f"**Template #2 Content**:\n"
            f"```\n{template2_content}\n```\n\n"
        )
    
    return JUDGER_INSTRUCTION_TEMPLATE.format(
        doi=doi,
        original_text=original_text,
        template1_output=template1_output,
        template2_output=template2_output,
        expert_annotations=expert_annotations,
        optional_template_content=optional_template_content
    )

def call_openai_judger(prompt, model_name="gpt-4o", temperature=0.3):
    """Call OpenAI API for LLM judging."""
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("Please set your OpenAI API key in the script before using the LLM judger.")
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": JUDGER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Track API usage
        API_USAGE_TRACKER['openai_calls'] += 1
        if hasattr(response, 'usage') and response.usage:
            API_USAGE_TRACKER['openai_request_tokens'] += response.usage.prompt_tokens
            API_USAGE_TRACKER['openai_response_tokens'] += response.usage.completion_tokens
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API for judger: {e}")
        return None

def extract_judger_decision(response):
    """Extract winner and reasoning from judger response."""
    try:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: look for JSON-like patterns
            json_match = re.search(r'\{\s*"winner".*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None, "Could not extract decision from response"
        
        decision_data = json.loads(json_str)
        winner = decision_data.get("winner", "").strip()
        reasoning = decision_data.get("reasoning", "").strip()
        
        # Validate winner format
        if winner not in ["#1", "#2"]:
            return None, f"Invalid winner format: {winner}"
        
        return winner, reasoning
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"
    except Exception as e:
        return None, f"Error extracting decision: {e}"

def judge_single_doi_duel(doi, minibatch_name, template1_version="iter1", template2_version="iter2", 
                         show_templates=False, duel_rounds=1):
    """Conduct LLM judging for a single DOI between two template versions."""
    
    sanitized_doi = sanitize_filename(doi)
    
    # Get original text from initial processing
    try:
        expert_path = Path(f"output/{minibatch_name}/init_prompt/{sanitized_doi}/JSON_expert.md")
        with open(expert_path, 'r', encoding='utf-8') as f:
            expert_content = f.read()
        expert_json_str = extract_json_from_content(expert_content)
        
        # Find original text from training files
        original_text = None
        training_files = get_training_files()
        for file_path in training_files:
            data = read_json_file(file_path)
            if data and data.get('DOI') == doi:
                original_text = data.get('Original_Text', '')
                break
        
        if not original_text:
            print(f"Could not find original text for DOI: {doi}")
            return None
            
    except Exception as e:
        print(f"Error reading expert data for {doi}: {e}")
        return None
    
    # Get template outputs
    template1_path = Path(f"output/{minibatch_name}/{template1_version}/{sanitized_doi}/JSON_output.md")
    template2_path = Path(f"output/{minibatch_name}/{template2_version}/{sanitized_doi}/JSON_output.md")
    
    if not template1_path.exists() or not template2_path.exists():
        print(f"Template outputs not found for {doi}")
        return None
    
    try:
        with open(template1_path, 'r', encoding='utf-8') as f:
            template1_output = f.read().strip()
        with open(template2_path, 'r', encoding='utf-8') as f:
            template2_output = f.read().strip()
    except Exception as e:
        print(f"Error reading template outputs for {doi}: {e}")
        return None
    
    # Get template contents if requested
    template1_content = ""
    template2_content = ""
    if show_templates:
        try:
            t1_path = Path(f"optimized_prompt/{minibatch_name}/Prompt_{minibatch_name}_{template1_version}.md")
            t2_path = Path(f"optimized_prompt/{minibatch_name}/Prompt_{minibatch_name}_{template2_version}.md")
            
            if t1_path.exists():
                with open(t1_path, 'r', encoding='utf-8') as f:
                    template1_content = f.read().strip()
            if t2_path.exists():
                with open(t2_path, 'r', encoding='utf-8') as f:
                    template2_content = f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read template contents: {e}")
    
    # Conduct multiple rounds if specified
    results = []
    for round_num in range(duel_rounds):
        print(f"    🥊 Round {round_num + 1}/{duel_rounds}")
        
        # Create judger prompt
        judger_prompt = create_llm_judger_prompt(
            doi=doi,
            original_text=original_text,
            template1_output=template1_output,
            template2_output=template2_output,
            expert_annotations=expert_json_str,
            show_templates=show_templates,
            template1_content=template1_content,
            template2_content=template2_content
        )
        
        # Call OpenAI judger
        judger_response = call_openai_judger(judger_prompt)
        if not judger_response:
            print(f"    ❌ Round {round_num + 1} failed")
            continue
        
        # Extract decision
        winner, reasoning = extract_judger_decision(judger_response)
        if winner:
            results.append({
                'round': round_num + 1,
                'winner': winner,
                'reasoning': reasoning,
                'full_response': judger_response
            })
            print(f"    ✅ Round {round_num + 1}: Winner {winner}")
        else:
            print(f"    ❌ Round {round_num + 1}: {reasoning}")
    
    if not results:
        return None
    
    # Determine overall winner for this DOI
    winner_counts = {"#1": 0, "#2": 0}
    for result in results:
        winner_counts[result['winner']] += 1
    
    overall_winner = "#1" if winner_counts["#1"] > winner_counts["#2"] else "#2"
    
    return {
        'doi': doi,
        'template1_version': template1_version,
        'template2_version': template2_version,
        'rounds': results,
        'winner_counts': winner_counts,
        'overall_winner': overall_winner,
        'total_rounds': len(results)
    }

def run_llm_judger_duel(minibatch_name, dois, template1_version="iter1", template2_version="iter2",
                       show_templates=False, duel_rounds=1, duel_set="minibatch"):
    """Run LLM judger dueling for the specified set of DOIs."""
    
    print(f"\n{'='*60}")
    print("🥊 LLM JUDGER DUELING SYSTEM")
    print(f"Template #1: {template1_version.upper()} vs Template #2: {template2_version.upper()}")
    print(f"Duel Set: {duel_set.upper()} | Rounds per DOI: {duel_rounds}")
    print(f"Show Template Content: {show_templates}")
    print(f"{'='*60}")
    
    duel_results = []
    
    for i, doi in enumerate(dois, 1):
        print(f"\n📄 DOI {i}/{len(dois)}: {doi}")
        
        result = judge_single_doi_duel(
            doi=doi,
            minibatch_name=minibatch_name,
            template1_version=template1_version,
            template2_version=template2_version,
            show_templates=show_templates,
            duel_rounds=duel_rounds
        )
        
        if result:
            duel_results.append(result)
            print(f"  🏆 Winner: Template {result['overall_winner']} ({result['winner_counts'][result['overall_winner']]}/{result['total_rounds']} rounds)")
        else:
            print("  ❌ Duel failed for this DOI")
    
    if not duel_results:
        print("\n❌ No successful duels conducted")
        return None
    
    # Calculate overall statistics
    overall_winner_counts = {"#1": 0, "#2": 0}
    for result in duel_results:
        overall_winner_counts[result['overall_winner']] += 1
    
    total_dois = len(duel_results)
    template1_wins = overall_winner_counts["#1"]
    template2_wins = overall_winner_counts["#2"]
    
    overall_champion = "#1" if template1_wins > template2_wins else "#2"
    champion_version = template1_version if overall_champion == "#1" else template2_version
    
    print(f"\n{'='*60}")
    print("🏆 LLM JUDGER DUEL RESULTS")
    print(f"{'='*60}")
    print(f"Template #1 ({template1_version.upper()}): {template1_wins}/{total_dois} DOIs ({template1_wins/total_dois*100:.1f}%)")
    print(f"Template #2 ({template2_version.upper()}): {template2_wins}/{total_dois} DOIs ({template2_wins/total_dois*100:.1f}%)")
    print(f"\n🎯 OVERALL CHAMPION: Template {overall_champion} ({champion_version.upper()})")
    print(f"   Victory Margin: {abs(template1_wins - template2_wins)}/{total_dois} DOIs")
    
    # Save results
    save_llm_judger_results(duel_results, minibatch_name, template1_version, template2_version, duel_set)
    
    return {
        'duel_results': duel_results,
        'overall_winner_counts': overall_winner_counts,
        'champion': overall_champion,
        'champion_version': champion_version,
        'total_dois': total_dois
    }

def save_llm_judger_results(duel_results, minibatch_name, template1_version, template2_version, duel_set):
    """Save LLM judger duel results."""
    output_dir = Path("output") / minibatch_name
    output_dir.mkdir(exist_ok=True)
    
    filename = f"LLM_Judger_Duel_{template1_version}_vs_{template2_version}_{duel_set}.json"
    filepath = output_dir / filename
    
    summary_data = {
        'metadata': {
            'minibatch_name': minibatch_name,
            'template1_version': template1_version,
            'template2_version': template2_version,
            'duel_set': duel_set,
            'total_dois': len(duel_results)
        },
        'results': duel_results
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 LLM Judger results saved to: {filepath}")
    except Exception as e:
        print(f"Error saving LLM judger results: {e}")

def get_dev_files(dev_path="split_datasets/fold_3/dev"):
    """Get list of JSON files from dev set."""
    try:
        dev_dir = Path(dev_path)
        if not dev_dir.exists():
            print(f"Dev directory not found: {dev_path}")
            return []
        
        json_files = list(dev_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in dev set")
        return json_files
    except Exception as e:
        print(f"Error accessing dev files: {e}")
        return []

def get_dois_from_files(json_files):
    """Extract DOIs from JSON files."""
    dois = []
    for file_path in json_files:
        data = read_json_file(file_path)
        if data and 'DOI' in data:
            dois.append(data['DOI'])
    return dois

def run_hierarchical_llm_duel(minibatch_name, dois, child1="iter1", child2="iter2", parent="init_prompt",
                             show_templates=False, duel_rounds=1, duel_set="minibatch"):
    """Run hierarchical LLM dueling: child vs child, then winner vs parent."""
    
    print(f"\n{'='*70}")
    print("🏆 HIERARCHICAL LLM JUDGER TOURNAMENT")
    print(f"Round 1: {child1.upper()} vs {child2.upper()} (Child Generation Duel)")
    print(f"Round 2: Winner vs {parent.upper()} (Child vs Parent Duel)")
    print(f"{'='*70}")
    
    # Phase 1: Child Generation Duel
    print(f"\n🥊 PHASE 1: CHILD GENERATION DUEL")
    print(f"{'='*50}")
    
    child_duel_results = run_llm_judger_duel(
        minibatch_name=minibatch_name,
        dois=dois,
        template1_version=child1,
        template2_version=child2,
        show_templates=show_templates,
        duel_rounds=duel_rounds,
        duel_set=duel_set
    )
    
    if not child_duel_results:
        print("❌ Child generation duel failed")
        return None
    
    # Determine winning child
    child_champion = child_duel_results['champion_version']
    child_victory_margin = abs(child_duel_results['overall_winner_counts']['#1'] - 
                              child_duel_results['overall_winner_counts']['#2'])
    
    # Handle tie case
    if child_victory_margin == 0:
        import random
        child_champion = random.choice([child1, child2])
        print(f"⚖️  TIE detected! Randomly selected {child_champion.upper()} as winning child")
    
    print(f"\n🎯 CHILD GENERATION WINNER: {child_champion.upper()}")
    
    # Phase 2: Child vs Parent Duel
    print(f"\n🥊 PHASE 2: CHILD VS PARENT DUEL")
    print(f"{child_champion.upper()} vs {parent.upper()}")
    print(f"{'='*50}")
    
    parent_duel_results = run_llm_judger_duel(
        minibatch_name=minibatch_name,
        dois=dois,
        template1_version=child_champion,
        template2_version=parent,
        show_templates=show_templates,
        duel_rounds=duel_rounds,
        duel_set=f"{duel_set}_final"
    )
    
    if not parent_duel_results:
        print("❌ Child vs parent duel failed")
        return None
    
    # Determine final champion
    final_champion_template = parent_duel_results['champion_version']
    final_victory_margin = abs(parent_duel_results['overall_winner_counts']['#1'] - 
                              parent_duel_results['overall_winner_counts']['#2'])
    
    # Handle tie case
    if final_victory_margin == 0:
        import random
        final_champion_template = random.choice([child_champion, parent])
        print(f"⚖️  TIE detected! Randomly selected {final_champion_template.upper()} as final champion")
    
    # Final Results Summary
    print(f"\n{'='*70}")
    print("🏆 HIERARCHICAL TOURNAMENT RESULTS")
    print(f"{'='*70}")
    print(f"📊 Phase 1 - Child Generation Duel:")
    print(f"   Winner: {child_champion.upper()} defeated {child2.upper() if child_champion == child1 else child1.upper()}")
    print(f"   Victory Margin: {child_victory_margin}/{len(dois)} DOIs")
    
    print(f"\n📊 Phase 2 - Child vs Parent Duel:")
    print(f"   Winner: {final_champion_template.upper()} defeated {parent.upper() if final_champion_template == child_champion else child_champion.upper()}")
    print(f"   Victory Margin: {final_victory_margin}/{len(dois)} DOIs")
    
    print(f"\n🎯 FINAL TOURNAMENT CHAMPION: {final_champion_template.upper()}")
    
    # Determine champion category
    if final_champion_template == parent:
        champion_category = "PARENT (Original Prompt)"
    else:
        champion_category = f"CHILD (Optimized Generation)"
    
    print(f"🏅 Champion Category: {champion_category}")
    
    # Save hierarchical results
    save_hierarchical_results(child_duel_results, parent_duel_results, 
                             child_champion, final_champion_template, 
                             minibatch_name, duel_set)
    
    return {
        'child_duel_results': child_duel_results,
        'parent_duel_results': parent_duel_results,
        'child_champion': child_champion,
        'final_champion': final_champion_template,
        'champion_category': champion_category,
        'total_dois': len(dois)
    }

def save_hierarchical_results(child_results, parent_results, child_champion, final_champion, minibatch_name, duel_set):
    """Save hierarchical duel results."""
    output_dir = Path("output") / minibatch_name
    output_dir.mkdir(exist_ok=True)
    
    filename = f"LLM_Hierarchical_Tournament_{duel_set}.json"
    filepath = output_dir / filename
    
    hierarchical_data = {
        'metadata': {
            'minibatch_name': minibatch_name,
            'duel_set': duel_set,
            'tournament_type': 'hierarchical'
        },
        'phase1_child_generation_duel': child_results,
        'phase2_child_vs_parent_duel': parent_results,
        'results_summary': {
            'child_champion': child_champion,
            'final_champion': final_champion,
            'champion_category': 'parent' if final_champion == 'init_prompt' else 'child'
        }
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(hierarchical_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Hierarchical tournament results saved to: {filepath}")
    except Exception as e:
        print(f"Error saving hierarchical results: {e}")

def determine_minibatch_winner(minibatch_name, dois, duel_mode="no_duel", winner_basis="hard_score", 
                              enforce_child_replacement=False, primary_metric="bertscore", 
                              committee_metrics="committee_default", llm_results=None, hierarchical_results=None):
    """Determine the winner for the minibatch based on specified criteria."""
    
    print(f"\n{'='*60}")
    print("🏆 DETERMINING MINIBATCH WINNER")
    print(f"Duel Mode: {duel_mode.upper()}")
    print(f"Winner Basis: {winner_basis.upper()}")
    print(f"Enforce Child Replacement: {enforce_child_replacement}")
    print(f"{'='*60}")
    
    # Get hard scores
    dueling_results = evaluate_dueling_scores(
        minibatch_name, dois, ["init_prompt", "iter1", "iter2"],
        primary_metric=primary_metric, 
        committee_metrics=committee_metrics
    )
    
    # Calculate version averages for hard scores
    version_averages = {}
    available_versions = list(set([version for doi in dueling_results for version in dueling_results[doi].keys()]))
    
    for version in available_versions:
        if not any(version in dueling_results[doi] for doi in dueling_results):
            continue
            
        # Calculate averages for each metric
        metrics = ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']
        averages = {}
        
        for metric in metrics:
            scores_list = [dueling_results[doi][version][metric] for doi in dueling_results if version in dueling_results[doi] and metric in dueling_results[doi][version]]
            averages[metric] = sum(scores_list) / len(scores_list) if scores_list else 0.0
        
        version_averages[version] = averages
    
    # Determine winner based on winner_basis
    if winner_basis == "hard_score":
        # Use hard scores to determine winner
        if version_averages:
            # Check if we should limit comparison based on LLM analysis
            eligible_versions = ["init_prompt", "iter1", "iter2"]
            
            # For mixed modes: only compare LLM-determined viable child vs parent
            if duel_mode == "child_only" and llm_results:
                # Child analysis mode: only compare LLM child winner vs parent
                child_champion = llm_results['champion_version']
                eligible_versions = ["init_prompt", child_champion]
                print(f"\n🔍 CHILD ANALYSIS MODE: Comparing {child_champion.upper()} (LLM winner) vs INIT_PROMPT")
                
            elif duel_mode == "child_parent" and hierarchical_results:
                # Full analysis mode: only compare hierarchical child winner vs parent
                child_champion = hierarchical_results['child_champion']
                eligible_versions = ["init_prompt", child_champion]
                print(f"\n🔍 HIERARCHICAL ANALYSIS MODE: Comparing {child_champion.upper()} (child champion) vs INIT_PROMPT")
            
            # Filter to eligible versions only
            eligible_averages = {v: version_averages[v] for v in eligible_versions if v in version_averages}
            
            if eligible_averages:
                primary_winner = max(eligible_averages.keys(), key=lambda v: eligible_averages[v].get(primary_metric, 0.0))
                committee_winner = max(eligible_averages.keys(), key=lambda v: eligible_averages[v]['committee'])
                
                print(f"\n📊 HARD SCORE COMPARISON (Eligible Versions Only):")
                for version in eligible_versions:
                    if version in version_averages:
                        avg = version_averages[version]
                        print(f"  {version:12} - {primary_metric.upper()}: {avg.get(primary_metric, 0.0):.4f}, Committee: {avg['committee']:.4f}")
                
                # Choose winner (prioritize committee score)
                hard_score_winner = committee_winner
                print(f"\n🎯 HARD SCORE WINNER (from eligible): {hard_score_winner.upper()}")
                
                # Apply enforcement rules
                if enforce_child_replacement and hard_score_winner == "init_prompt":
                    # Force child replacement - choose best eligible child
                    child_scores = {v: eligible_averages[v]['committee'] for v in eligible_averages if v.startswith('iter')}
                    if child_scores:
                        forced_winner = max(child_scores.keys(), key=lambda v: child_scores[v])
                        print(f"⚡ ENFORCING CHILD REPLACEMENT: {forced_winner.upper()} (was {hard_score_winner.upper()})")
                        final_winner = forced_winner
                    else:
                        final_winner = hard_score_winner
                else:
                    final_winner = hard_score_winner
            else:
                print("❌ No eligible versions for hard score comparison")
                final_winner = "init_prompt"  # Fallback
        else:
            print("❌ No hard score results available")
            final_winner = "init_prompt"  # Fallback
            
    elif winner_basis == "duel_result":
        # Use duel results to determine winner
        if duel_mode == "child_only" and llm_results:
            champion_version = llm_results['champion_version']
            print(f"\n🥊 CHILD DUEL WINNER: {champion_version.upper()}")
            
            if enforce_child_replacement:
                final_winner = champion_version  # Already a child
                print(f"⚡ CHILD REPLACEMENT ENFORCED: {final_winner.upper()}")
            else:
                final_winner = champion_version
                
        elif duel_mode == "child_parent" and hierarchical_results:
            final_champion = hierarchical_results['final_champion']
            print(f"\n🏆 HIERARCHICAL DUEL WINNER: {final_champion.upper()}")
            
            if enforce_child_replacement and final_champion == "init_prompt":
                # Force child replacement
                child_champion = hierarchical_results['child_champion']
                print(f"⚡ ENFORCING CHILD REPLACEMENT: {child_champion.upper()} (was {final_champion.upper()})")
                final_winner = child_champion
            else:
                final_winner = final_champion
        else:
            print("❌ No duel results available, falling back to hard scores")
            return determine_minibatch_winner(minibatch_name, dois, "no_duel", "hard_score", 
                                            enforce_child_replacement, primary_metric, committee_metrics)
    
    print(f"\n🎯 FINAL MINIBATCH WINNER: {final_winner.upper()}")
    
    # Save winner decision
    save_winner_decision(minibatch_name, final_winner, winner_basis, duel_mode, enforce_child_replacement, 
                        version_averages, llm_results, hierarchical_results)
    
    return final_winner

def save_winner_decision(minibatch_name, winner, winner_basis, duel_mode, enforce_child_replacement,
                        version_averages, llm_results, hierarchical_results):
    """Save the winner decision and reasoning."""
    output_dir = Path("output") / minibatch_name
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / "Winner_Decision.json"
    
    decision_data = {
        'metadata': {
            'minibatch_name': minibatch_name,
            'winner_basis': winner_basis,
            'duel_mode': duel_mode,
            'enforce_child_replacement': enforce_child_replacement
        },
        'final_winner': winner,
        'hard_score_averages': version_averages,
        'llm_duel_results': llm_results,
        'hierarchical_results': hierarchical_results
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(decision_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Winner decision saved to: {filepath}")
    except Exception as e:
        print(f"Error saving winner decision: {e}")

def get_winner_prompt_path(minibatch_name, winner_version, current_prompt_file="human_init_prompt.md"):
    """Get the path to the winner prompt file."""
    if winner_version == "init_prompt":
        # Use the current prompt file that was being used in this round
        # This could be the original or an inherited prompt from previous rounds
        return current_prompt_file
    else:
        # Get optimized prompt path
        optimized_path = Path(f"optimized_prompt/{minibatch_name}/Prompt_{minibatch_name}_{winner_version}.md")
        if optimized_path.exists():
            return str(optimized_path)
        else:
            print(f"Warning: Optimized prompt not found at {optimized_path}")
            return current_prompt_file  # Fallback to current prompt

def copy_winner_prompt_for_next_round(winner_version, current_minibatch, next_minibatch, current_prompt_file="human_init_prompt.md"):
    """Copy the winner prompt to be used as init prompt for next round."""
    winner_path = get_winner_prompt_path(current_minibatch, winner_version, current_prompt_file)
    
    # Create next round init prompt
    next_prompt_path = f"training_round_prompts/init_prompt_{next_minibatch}.md"
    
    # Create directory
    Path("training_round_prompts").mkdir(exist_ok=True)
    
    try:
        with open(winner_path, 'r', encoding='utf-8') as f:
            winner_content = f.read()
        
        with open(next_prompt_path, 'w', encoding='utf-8') as f:
            f.write(winner_content)
        
        print(f"📋 Winner prompt copied: {winner_path} -> {next_prompt_path}")
        return next_prompt_path
    except Exception as e:
        print(f"Error copying winner prompt: {e}")
        return "human_init_prompt.md"  # Fallback

def save_training_summary(training_history, args):
    """Save a comprehensive training summary."""
    summary_path = Path("training_summary.json")
    
    summary_data = {
        'metadata': {
            'total_rounds': args.training_rounds,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'learning_rate': args.learning_rate,
            'duel_mode': args.duel_mode,
            'winner_basis': args.winner_basis,
            'enforce_child_replacement': args.enforce_child_replacement,
            'include_hardscore_in_summarizer': args.include_hardscore_in_summarizer,
            'primary_metric': args.primary_metric,
            'committee_metrics': args.committee_metrics
        },
        'training_history': training_history,
        'final_results': {
            'final_champion': training_history[-1]['winner'] if training_history else "init_prompt",
            'final_minibatch': training_history[-1]['minibatch_name'] if training_history else args.minibatch_name,
            'total_rounds_completed': len(training_history)
        }
    }
    
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"📋 Training summary saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving training summary: {e}")

def save_final_results(final_winner, final_minibatch, training_history, args, final_prompt_file):
    """Save final optimized prompt and history in separate folder."""
    final_dir = Path("final_results")
    final_dir.mkdir(exist_ok=True)
    
    try:
        # Copy final optimized prompt
        if final_winner != "init_prompt":
            source_prompt = Path(f"optimized_prompt/{final_minibatch}/Prompt_{final_minibatch}_{final_winner}.md")
            if source_prompt.exists():
                with open(source_prompt, 'r', encoding='utf-8') as f:
                    final_prompt = f.read()
                with open(final_dir / "final_optimized_prompt.md", 'w', encoding='utf-8') as f:
                    f.write(final_prompt)
            else:
                print(f"Warning: Optimized prompt not found at {source_prompt}, using final prompt file")
                with open(final_prompt_file, 'r', encoding='utf-8') as f:
                    final_prompt = f.read()
                with open(final_dir / "final_optimized_prompt.md", 'w', encoding='utf-8') as f:
                    f.write(final_prompt)
        else:
            # Copy the actual final prompt file (could be evolved, not necessarily original)
            print(f"Final winner is INIT_PROMPT - using actual prompt file: {final_prompt_file}")
            with open(final_prompt_file, 'r', encoding='utf-8') as f:
                final_prompt = f.read()
            with open(final_dir / "final_optimized_prompt.md", 'w', encoding='utf-8') as f:
                f.write(final_prompt)
        
        # Copy training summary
        if Path("training_summary.json").exists():
            with open("training_summary.json", 'r', encoding='utf-8') as f:
                summary_data = f.read()
            with open(final_dir / "training_summary.json", 'w', encoding='utf-8') as f:
                f.write(summary_data)
        
        # Create intuitive training summary table
        create_training_summary_table(training_history, args, final_dir)
        
        # Save API usage report
        save_api_usage_report(final_dir, training_history, args)
        
        print(f"📁 Final results saved to: {final_dir}/")
    except Exception as e:
        print(f"Error saving final results: {e}")

def save_api_usage_report(final_dir, training_history, args):
    """Save API usage statistics report."""
    # Calculate estimated costs (rough estimates based on current pricing)
    openai_request_cost = API_USAGE_TRACKER['openai_request_tokens'] * (1.1/1000000)  # ~$1.1/1M tokens
    openai_response_cost = API_USAGE_TRACKER['openai_response_tokens'] * (4.4/1000000)  # ~$4.4/1M tokens
    total_openai_cost = openai_request_cost + openai_response_cost
    
    report_content = f"""# 🔄 API Usage Report

## 📊 API Call Statistics

### Local Ollama Usage
- **Total Calls**: {API_USAGE_TRACKER['ollama_calls']:,}
- **Response Tokens**: {API_USAGE_TRACKER['ollama_response_tokens']:,}
- **Average Tokens per Call**: {API_USAGE_TRACKER['ollama_response_tokens'] // max(1, API_USAGE_TRACKER['ollama_calls']):,}

### OpenAI Cloud Usage  
- **Total Calls**: {API_USAGE_TRACKER['openai_calls']:,}
- **Request Tokens (Sent)**: {API_USAGE_TRACKER['openai_request_tokens']:,}
- **Response Tokens (Received)**: {API_USAGE_TRACKER['openai_response_tokens']:,}
- **Total Tokens**: {API_USAGE_TRACKER['openai_request_tokens'] + API_USAGE_TRACKER['openai_response_tokens']:,}

### Cost Estimation (USD)
- **Request Cost**: ${openai_request_cost:.4f}
- **Response Cost**: ${openai_response_cost:.4f}
- **Total OpenAI Cost**: ${total_openai_cost:.4f}

## 🧠 Usage Breakdown by Component

### Training Configuration
- **Training Rounds**: {len(training_history)}
- **Strategy**: {args.duel_mode} mode with {args.winner_basis} basis
- **Batch Size**: {args.batch_size} papers per round
- **Iterations**: {args.iterations} per round

### Expected Usage Pattern
- **Ollama**: Scientific paper extraction (main processing)
- **OpenAI o4-mini**: Summarizer + Updater optimization
- **OpenAI GPT-4o**: LLM judger evaluation (if enabled)

## 💡 Efficiency Metrics
- **Papers Processed**: ~{len(training_history) * args.batch_size}
- **Ollama Calls per Paper**: {API_USAGE_TRACKER['ollama_calls'] / max(1, len(training_history) * args.batch_size):.1f}
- **OpenAI Calls per Round**: {API_USAGE_TRACKER['openai_calls'] / max(1, len(training_history)):.1f}
- **Total Tokens per Round**: {(API_USAGE_TRACKER['openai_request_tokens'] + API_USAGE_TRACKER['openai_response_tokens']) / max(1, len(training_history)):.0f}

---
*Generated automatically by Multi-Minibatch Training System*
"""
    
    try:
        with open(final_dir / "API_Usage_Report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📊 API usage report saved to: {final_dir}/API_Usage_Report.md")
    except Exception as e:
        print(f"Error saving API usage report: {e}")

def create_training_summary_table(training_history, args, final_dir):
    """Create an intuitive markdown table summarizing the training process."""
    
    # Read hard scores for each round to get performance data
    performance_data = {}
    for entry in training_history:
        minibatch_name = entry['minibatch_name']
        hard_score_file = Path(f"output/{minibatch_name}/hard_score.json")
        winner_file = Path(f"output/{minibatch_name}/Winner_Decision.json")
        
        if hard_score_file.exists():
            try:
                with open(hard_score_file, 'r') as f:
                    scores = json.load(f)
                performance_data[minibatch_name] = scores
            except:
                performance_data[minibatch_name] = {}
        
        if winner_file.exists():
            try:
                with open(winner_file, 'r') as f:
                    winner_data = json.load(f)
                if minibatch_name not in performance_data:
                    performance_data[minibatch_name] = {}
                performance_data[minibatch_name]['winner_info'] = winner_data
            except:
                pass
    
    # Create the markdown table
    markdown_content = f"""# 🧠 Multi-Minibatch Training Summary

## 📋 Training Configuration
- **Strategy**: {args.duel_mode.upper()} mode with {args.winner_basis.upper()} basis
- **Total Rounds**: {args.training_rounds}
- **Batch Size**: {args.batch_size} papers per round
- **Learning Rate**: {args.learning_rate}
- **Iterations**: {args.iterations} optimization cycles per round
- **Primary Metric**: {args.primary_metric.upper()}
- **Force Child Replacement**: {'✅ Yes' if args.enforce_child_replacement else '❌ No'}
- **Include Hard Score in Summarizer**: {'✅ Yes' if args.include_hardscore_in_summarizer else '❌ No'}

## 🏆 Training Journey

| Round | Minibatch | Starting Prompt | Winner | Action | Performance | Evolution Status |
|-------|-----------|----------------|--------|--------|-------------|------------------|
"""
    
    for i, entry in enumerate(training_history, 1):
        round_num = entry['round']
        minibatch_name = entry['minibatch_name']
        winner = entry['winner']
        starting_prompt = entry['prompt_used']
        
        # Get performance data
        scores_data = performance_data.get(minibatch_name, {})
        avg_scores = scores_data.get('winner_info', {}).get('hard_score_averages', {})
        
        # Determine starting prompt type
        if "human_init_prompt.md" in starting_prompt:
            prompt_emoji = "🌱"
            prompt_desc = "Original Human"
        elif "training_round_prompts" in starting_prompt:
            prompt_emoji = "🧬"
            prompt_desc = "Inherited Winner"
        else:
            prompt_emoji = "📝"
            prompt_desc = "Custom"
        
        # Determine winner emoji and description
        if winner == "init_prompt":
            winner_emoji = "🏠"
            winner_desc = "INIT_PROMPT"
            if i == 1:
                evolution_status = "🔴 Original Wins"
            else:
                evolution_status = "🔄 Inherited Wins"
        elif winner.startswith("iter"):
            winner_emoji = "🚀"
            winner_desc = winner.upper()
            evolution_status = "🟢 Optimized Wins"
        else:
            winner_emoji = "❓"
            winner_desc = winner.upper()
            evolution_status = "❓ Unknown"
        
        # Get performance summary
        if avg_scores and winner in avg_scores:
            winner_scores = avg_scores[winner]
            primary_score = winner_scores.get(args.primary_metric, 0.0)
            committee_score = winner_scores.get('committee', 0.0)
            performance = f"🎯 {args.primary_metric.upper()}: {primary_score:.3f}<br/>🏛️ Committee: {committee_score:.3f}"
        else:
            performance = "📊 Scores pending"
        
        # Determine action taken
        if i < len(training_history):
            if winner == "init_prompt":
                action = f"📋 Copy inherited prompt to Round {i+1}"
            else:
                action = f"📋 Copy {winner.upper()} to Round {i+1}"
        else:
            action = f"🏁 Final champion: {winner.upper()}"
        
        # Add row to table
        markdown_content += f"| **{round_num}** | `{minibatch_name}` | {prompt_emoji} {prompt_desc} | {winner_emoji} **{winner_desc}** | {action} | {performance} | {evolution_status} |\n"
    
    # Add evolution summary
    markdown_content += f"""
## 📈 Evolution Summary

"""
    
    # Count evolution types
    original_wins = sum(1 for entry in training_history if entry['winner'] == 'init_prompt')
    optimized_wins = sum(1 for entry in training_history if entry['winner'].startswith('iter'))
    
    markdown_content += f"""### 🏆 Winner Distribution
- **🏠 Original/Inherited Prompt Wins**: {original_wins}/{len(training_history)} rounds ({original_wins/len(training_history)*100:.1f}%)
- **🚀 Optimized Prompt Wins**: {optimized_wins}/{len(training_history)} rounds ({optimized_wins/len(training_history)*100:.1f}%)

### 🧬 Prompt Evolution Chain
"""
    
    for i, entry in enumerate(training_history, 1):
        if i == 1:
            markdown_content += f"1. **Round {entry['round']}**: Started with 🌱 Original → Winner: {entry['winner'].upper()}\n"
        else:
            prev_winner = training_history[i-2]['winner']
            markdown_content += f"{i}. **Round {entry['round']}**: Started with 🧬 {prev_winner.upper()} → Winner: {entry['winner'].upper()}\n"
    
    final_champion = training_history[-1]['winner']
    if final_champion == 'init_prompt':
        champion_type = "🏠 Inherited Prompt (Original DNA)"
    else:
        champion_type = f"🚀 Optimized Prompt ({final_champion.upper()})"
    
    markdown_content += f"""
### 🎯 Final Outcome
**🏆 FINAL CHAMPION**: {champion_type}

**🧪 Training Effectiveness**: 
- Strategy: `{args.duel_mode}` mode with `{args.winner_basis}` basis
- Evolution: {optimized_wins} out of {len(training_history)} rounds favored optimization
- Consistency: {'✅ Stable evolution' if optimized_wins > original_wins else '⚠️ Conservative evolution' if original_wins > optimized_wins else '⚖️ Balanced evolution'}

**📊 Methodology Notes**:
- Each round processes {args.batch_size} random papers from training set
- Winner selection based on {args.primary_metric.upper()} and committee metrics
- Learning rate {args.learning_rate} controls optimization aggressiveness
{f"- 🔄 Child replacement enforced regardless of performance" if args.enforce_child_replacement else "- 🎯 Performance-based winner selection"}
{f"- 📈 Hard scores included in optimization feedback" if args.include_hardscore_in_summarizer else "- 🎨 Pure qualitative optimization feedback"}

---
*Generated automatically by Multi-Minibatch Training System*
"""
    
    # Save the markdown file
    try:
        with open(final_dir / "Training_Journey_Summary.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"📊 Training journey summary saved to: {final_dir}/Training_Journey_Summary.md")
    except Exception as e:
        print(f"Error saving training journey summary: {e}")

# Global variables for logging
CLUSTER_LOG_DIR = Path("cluster_logs")
CLUSTER_LOG_FILE = None

def setup_cluster_logging(base_name="training"):
    """Setup cluster logging with timestamp."""
    global CLUSTER_LOG_FILE
    CLUSTER_LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CLUSTER_LOG_FILE = CLUSTER_LOG_DIR / f"{base_name}_progress_{timestamp}.md"
    
    # Initialize log file
    with open(CLUSTER_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# Training Progress Log\n\n")
        f.write(f"**Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    print(f"📝 Cluster logging enabled: {CLUSTER_LOG_FILE}")

def log_to_cluster(message, level="INFO"):
    """Log message to cluster log file."""
    if CLUSTER_LOG_FILE is None:
        return
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        with open(CLUSTER_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"**{timestamp}** [{level}] {message}\n\n")
    except Exception as e:
        print(f"Warning: Could not write to cluster log: {e}")

def save_checkpoint(training_history, current_round, args, current_prompt_file):
    """Save training checkpoint."""
    checkpoint_data = {
        'training_history': training_history,
        'current_round': current_round,
        'args': vars(args),
        'current_prompt_file': current_prompt_file,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_file = Path("training_checkpoint.json")
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        log_to_cluster(f"💾 Checkpoint saved at round {current_round}")
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")

def load_checkpoint():
    """Load training checkpoint if exists."""
    checkpoint_file = Path("training_checkpoint.json")
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        print(f"📂 Found checkpoint from {checkpoint_data['timestamp']}")
        return checkpoint_data
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None

def cleanup_checkpoint():
    """Remove checkpoint file after successful completion."""
    checkpoint_file = Path("training_checkpoint.json")
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            print("🗑️ Checkpoint file cleaned up")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint: {e}")

def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(description='Process scientific papers with minibatch optimization')
    parser.add_argument('--batch-size', type=int, default=3, help='Number of papers in minibatch (default: 2)')
    parser.add_argument('--minibatch-name', type=str, default='minibatch', help='Name of the minibatch (default: minibatch)')
    parser.add_argument('--prompt-file', type=str, default='human_init_prompt.md', help='Path to prompt file (default: human_init_prompt.md)')
    parser.add_argument('--train-path', type=str, default='split_datasets/fold_3/train', help='Path to training set (default: split_datasets/fold_3/train)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling (default: 42)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for OpenAI API calls (default: 1.0)')
    parser.add_argument('--iterations', type=int, default=2, help='Number of summarizer-updater iterations (default: 2)')
    parser.add_argument('--primary-metric', type=str, default='bertscore', 
                       choices=['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee'],
                       help='Primary metric for evaluation (default: bertscore)')
    parser.add_argument('--committee-metrics', type=str, default='committee_default',
                       help='Committee metrics: "committee_default" (bertscore,rouge_1,meteor), "all", or comma-separated list (e.g., "bleu,meteor,exact_match")')
    
    # Model selection
    parser.add_argument('--model', type=str, default='deepseek-r1:14b',
                       help='Ollama model to use for processing (default: deepseek-r1:14b)')
    
    # API selection for summarizer and updater
    parser.add_argument('--summarizer-api', type=str, default='openai', 
                       choices=['openai', 'ollama'],
                       help='API to use for summarizer analysis: openai or ollama (default: openai)')
    parser.add_argument('--updater-api', type=str, default='openai',
                       choices=['openai', 'ollama'], 
                       help='API to use for updater optimization: openai or ollama (default: openai)')
    parser.add_argument('--meta-model', type=str, default='deepseek-r1:14b',
                       help='Ollama model to use for summarizer/updater when using ollama API (default: deepseek-r1:14b)')
    
    # LLM Judger arguments
    parser.add_argument('--llm-judger', action='store_true', 
                       help='Enable LLM judger evaluation (default: False)')
    parser.add_argument('--judger-duel-set', type=str, default='minibatch', 
                       choices=['minibatch', 'dev', 'both'],
                       help='Dataset for LLM judger dueling: minibatch, dev, or both (default: minibatch)')
    parser.add_argument('--judger-rounds', type=int, default=1,
                       help='Number of judging rounds per DOI for robustness (default: 1)')
    parser.add_argument('--judger-show-templates', action='store_true',
                       help='Show template content to LLM judger (default: False)')
    parser.add_argument('--judger-template1', type=str, default='iter1',
                       help='First template version for dueling (default: iter1)')
    parser.add_argument('--judger-template2', type=str, default='iter2',
                       help='Second template version for dueling (default: iter2)')
    parser.add_argument('--dev-path', type=str, default='split_datasets/fold_3/dev',
                       help='Path to dev set for LLM judger evaluation (default: split_datasets/fold_3/dev)')
    parser.add_argument('--child-parent-duel', action='store_true',
                       help='Enable child vs parent duel after child generation duel (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.3,
                       help='Learning rate for prompt optimization: controls change magnitude (0.1=conservative, 0.5=moderate, 1.0=aggressive) (default: 0.3)')
    
    # Multi-minibatch training
    parser.add_argument('--training-rounds', type=int, default=1,
                       help='Number of minibatch training rounds (default: 1, use 20-30 for full training)')
    
    # Winner selection framework
    parser.add_argument('--duel-mode', type=str, default='no_duel', 
                       choices=['no_duel', 'child_only', 'child_parent'],
                       help='Duel mode: no_duel (hard scores only), child_only (child duel), child_parent (hierarchical) (default: no_duel)')
    parser.add_argument('--enforce-child-replacement', action='store_true',
                       help='Force child replacement even if parent performs better (default: False)')
    parser.add_argument('--winner-basis', type=str, default='hard_score',
                       choices=['hard_score', 'duel_result'],
                       help='Basis for winner selection: hard_score or duel_result (default: hard_score)')
    parser.add_argument('--include-hardscore-in-summarizer', action='store_true',
                       help='Include hard score results in summarizer feedback (TextGrad baseline) (default: False)')
    
    # Resume functionality
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh training, ignore any existing checkpoint (default: False)')
    
    args = parser.parse_args()
    
    # Check for existing checkpoint
    checkpoint = None
    if not args.no_resume:
        checkpoint = load_checkpoint()
    
    if checkpoint:
        print(f"\n🔄 RESUMING TRAINING FROM CHECKPOINT")
        print(f"Previous run stopped at round {checkpoint['current_round']}")
        
        # Restore state
        training_history = checkpoint['training_history']
        start_round = checkpoint['current_round']
        current_prompt_file = checkpoint['current_prompt_file']
        
        # Verify checkpoint args match current args (key parameters)
        checkpoint_args = checkpoint['args']
        critical_params = ['training_rounds', 'batch_size', 'minibatch_name', 'model', 'seed']
        
        mismatch = False
        for param in critical_params:
            if getattr(args, param) != checkpoint_args.get(param):
                print(f"⚠️  Parameter mismatch: {param} = {getattr(args, param)} (current) vs {checkpoint_args.get(param)} (checkpoint)")
                mismatch = True
        
        if mismatch:
            response = input("Continue with current parameters? (y/n): ")
            if response.lower() != 'y':
                print("Exiting. Use --no-resume to start fresh.")
                sys.exit(1)
        
        log_to_cluster(f"🔄 Resumed training from checkpoint at round {start_round}")
        
    else:
        print(f"\n🚀 STARTING FRESH TRAINING")
        training_history = []
        start_round = 1
        current_prompt_file = args.prompt_file
    
    # Setup cluster logging
    setup_cluster_logging(f"training_{args.minibatch_name}")
    
    # Logical constraint: child_parent duel mode automatically uses duel_result
    if args.duel_mode == "child_parent" and args.winner_basis == "hard_score":
        print("⚡ Auto-correcting: child_parent duel mode requires duel_result winner basis")
        args.winner_basis = "duel_result"
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Check if prompt file exists
    if not os.path.exists(current_prompt_file):
        print(f"Error: Prompt file '{current_prompt_file}' not found.")
        sys.exit(1)
    
    # Get training files
    training_files = get_training_files(args.train_path)
    if not training_files:
        print("No training files found!")
        sys.exit(1)
    
    # Sample minibatch for first round (if starting fresh)
    if start_round == 1:
        minibatch_files = sample_minibatch(training_files, args.batch_size)
    
    print(f"\nTraining Configuration:")
    print(f"  Base Name: {args.minibatch_name}")
    print(f"  Training Rounds: {args.training_rounds}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Model: {args.model}")
    print(f"  Seed: {args.seed}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Iterations: {args.iterations}")
    print(f"\nAPI Configuration:")
    print(f"  Summarizer API: {args.summarizer_api}")
    print(f"  Updater API: {args.updater_api}")
    print(f"  Meta Model (Ollama): {args.meta_model}")
    print(f"\nWinner Selection Framework:")
    print(f"  Duel Mode: {args.duel_mode}")
    print(f"  Winner Basis: {args.winner_basis}")
    print(f"  Enforce Child Replacement: {args.enforce_child_replacement}")
    print(f"  Include Hard Score in Summarizer: {args.include_hardscore_in_summarizer}")
    
    if start_round == 1:
        print(f"\nRound 1 files selected:")
        for i, file_path in enumerate(minibatch_files, 1):
            print(f"    {i}. {file_path.name}")
    
    if checkpoint:
        log_to_cluster(f"🔄 Resumed training from round {start_round}/{args.training_rounds}")
    else:
        log_to_cluster(f"🚀 Training started with {args.training_rounds} rounds, batch size {args.batch_size}, model {args.model}")
    
    # Multi-minibatch training loop
    print(f"\n{'='*60}")
    if checkpoint:
        print(f"🔄 RESUMING MULTI-MINIBATCH TRAINING FROM ROUND {start_round}")
    else:
        print("🚀 STARTING MULTI-MINIBATCH TRAINING")
    print(f"Training Rounds: {start_round}-{args.training_rounds}")
    print(f"{'='*60}")
    
    for round_num in range(start_round, args.training_rounds + 1):
        print(f"\n{'='*80}")
        print(f"🔄 TRAINING ROUND {round_num}/{args.training_rounds}")
        print(f"{'='*80}")
        
        log_to_cluster(f"🔄 Starting training round {round_num}/{args.training_rounds}")
        
        # Generate unique minibatch name for this round
        current_minibatch_name = f"{args.minibatch_name}{round_num}" if args.training_rounds > 1 else args.minibatch_name
        
        # Sample new minibatch for this round (unless it's round 1 and starting fresh)
        if round_num == 1 and start_round == 1:
            current_minibatch_files = minibatch_files
        else:
            # Re-seed for this specific round to ensure reproducibility
            random.seed(args.seed + round_num - 1)
            current_minibatch_files = sample_minibatch(training_files, args.batch_size)
            print(f"\nRound {round_num} files:")
            for i, file_path in enumerate(current_minibatch_files, 1):
                print(f"    {i}. {file_path.name}")
        
        print(f"\nUsing prompt: {current_prompt_file}")
    
        # Process the minibatch
        winner = process_minibatch(
            current_minibatch_files, 
            current_prompt_file, 
            current_minibatch_name, 
            args.temperature, 
            args.iterations, 
            args.primary_metric, 
            args.committee_metrics,
            args.llm_judger,
            args.judger_duel_set,
            args.judger_rounds,
            args.judger_show_templates,
            args.judger_template1,
            args.judger_template2,
            args.dev_path,
            args.child_parent_duel,
            args.learning_rate,
            args.duel_mode,
            args.winner_basis,
            args.enforce_child_replacement,
            args.include_hardscore_in_summarizer,
            args.model,
            args.summarizer_api,
            args.updater_api,
            args.meta_model
        )
        
        if winner:
            print(f"\n✅ Round {round_num} completed successfully!")
            print(f"🏆 Round {round_num} winner: {winner.upper()}")
            
            log_to_cluster(f"✅ Round {round_num} completed successfully - Winner: {winner.upper()}")
            
            # Record training history
            training_history.append({
                'round': round_num,
                'minibatch_name': current_minibatch_name,
                'winner': winner,
                'prompt_used': current_prompt_file
            })
            
            # Prepare for next round
            if round_num < args.training_rounds:
                next_minibatch_name = f"{args.minibatch_name}{round_num + 1}"
                current_prompt_file = copy_winner_prompt_for_next_round(winner, current_minibatch_name, next_minibatch_name, current_prompt_file)
                print(f"📋 Winner prompt prepared for next round: {current_prompt_file}")
                
                # Save checkpoint before next round
                save_checkpoint(training_history, round_num + 1, args, current_prompt_file)
        else:
            print(f"\n❌ Round {round_num} failed!")
            log_to_cluster(f"❌ Round {round_num} failed!", "ERROR")
            sys.exit(1)
    
    # Training completed successfully
    cleanup_checkpoint()
    
    print(f"\n{'='*80}")
    print("🎊 MULTI-MINIBATCH TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    # Print training summary
    print(f"\n📊 TRAINING HISTORY:")
    for entry in training_history:
        print(f"  Round {entry['round']}: {entry['winner'].upper()} ({entry['minibatch_name']})")
    
    final_winner = training_history[-1]['winner'] if training_history else "init_prompt"
    final_minibatch = training_history[-1]['minibatch_name'] if training_history else args.minibatch_name
    print(f"\n🏆 FINAL CHAMPION: {final_winner.upper()}")
    print(f"📁 Check all outputs in: ./output/")
    print(f"📝 Final optimized prompt: ./optimized_prompt/{final_minibatch}/Prompt_{final_minibatch}_{final_winner}.md" if final_winner != "init_prompt" else f"Final prompt: {args.prompt_file}")
    
    log_to_cluster(f"🎊 Training completed! Final champion: {final_winner.upper()}")
    
    # Save training summary
    save_training_summary(training_history, args)
    
    # Save final results in separate folder
    final_prompt_file = training_history[-1]['prompt_used'] if training_history else args.prompt_file
    save_final_results(final_winner, final_minibatch, training_history, args, final_prompt_file)
    
    # Print API usage summary
    print(f"\n📊 API USAGE SUMMARY:")
    print(f"   Ollama calls: {API_USAGE_TRACKER['ollama_calls']:,}")
    print(f"   OpenAI calls: {API_USAGE_TRACKER['openai_calls']:,}")
    print(f"   Total tokens: {API_USAGE_TRACKER['openai_request_tokens'] + API_USAGE_TRACKER['openai_response_tokens']:,}")
    
    log_to_cluster(f"📊 Final API usage - Ollama: {API_USAGE_TRACKER['ollama_calls']:,} calls, OpenAI: {API_USAGE_TRACKER['openai_calls']:,} calls")
    
    print(f"\n🎉 All {args.training_rounds} training rounds completed successfully!")

if __name__ == "__main__":
    main() 