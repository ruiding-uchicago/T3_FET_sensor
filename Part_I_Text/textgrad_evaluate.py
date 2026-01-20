#!/usr/bin/env python3
"""
Script to evaluate prompt performance on test set.
Compares original human prompt vs final optimized prompt using comprehensive metrics.
"""

import json
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict
import ollama
from datetime import datetime

# FORCE OFFLINE MODE - prevent NLTK from trying to download anything
os.environ['NLTK_OFFLINE'] = '1'
import nltk
# Disable NLTK downloads completely
nltk.download = lambda *args, **kwargs: False

# Real-time logging setup
LOG_FILE = None

def log_print(*args, **kwargs):
    """Print and simultaneously write to log file."""
    import builtins
    # Regular print
    builtins.print(*args, **kwargs)
    
    # Write to log file if initialized
    if LOG_FILE:
        try:
            message = ' '.join(str(arg) for arg in args)
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
                f.flush()  # Ensure immediate write
        except Exception as e:
            builtins.print(f"Warning: Could not write to log file: {e}")

def initialize_logging(output_name="test_eval"):
    """Initialize real-time logging to markdown file."""
    global LOG_FILE
    log_dir = Path("cluster_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = log_dir / f"{output_name}_progress_{timestamp}.md"
    
    # Write header
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# Test Set Evaluation Progress Log\n\n")
            f.write(f"**Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Output Name**: {output_name}\n\n")
            f.write("## Progress Log\n\n")
        print(f"📝 Real-time logging initialized: {LOG_FILE}")
    except Exception as e:
        print(f"Warning: Could not initialize log file: {e}")
        LOG_FILE = None

# Replace print with log_print globally for this script
print = log_print

# Import necessary functions from textgrad_train.py
sys.path.append('.')

# Import functions from the other script
import importlib.util
spec = importlib.util.spec_from_file_location("textgrad_train", "textgrad_train.py")
textgrad_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(textgrad_train)

# Extract the functions we need
read_json_file = textgrad_train.read_json_file
read_human_prompt = textgrad_train.read_human_prompt
save_response = textgrad_train.save_response
save_expert_data = textgrad_train.save_expert_data
calculate_all_scores = textgrad_train.calculate_all_scores
calculate_committee_score = textgrad_train.calculate_committee_score
extract_json_from_content = textgrad_train.extract_json_from_content
sanitize_filename = textgrad_train.sanitize_filename
clean_response = textgrad_train.clean_response
API_USAGE_TRACKER = textgrad_train.API_USAGE_TRACKER

def call_ollama_fast(prompt, model="deepseek-r1:14b"):
    """Call Ollama using pip package for better performance."""
    try:
        # Configure Ollama client with custom host if provided
        ollama_host = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')
        client = ollama.Client(host=f"http://{ollama_host}")
        
        response = client.generate(
            model=model,
            prompt=prompt
        )
        
        response_text = response.get('response', '')
        
        # Track API usage (compatible with existing tracker)
        API_USAGE_TRACKER['ollama_calls'] += 1
        if response_text:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            API_USAGE_TRACKER['ollama_response_tokens'] += len(response_text) // 4
        
        return response_text
    except Exception as e:
        print(f"Error calling Ollama (pip package): {e}")
        return None

def calculate_bertscore_batch(text_pairs):
    """Batch BERTScore calculation - much faster than individual calls."""
    try:
        from bert_score import score
        import torch
        
        if not text_pairs:
            return []
        
        # Separate references and candidates
        references = []
        candidates = []
        valid_indices = []
        
        for i, (ref, cand) in enumerate(text_pairs):
            if ref and cand and ref.strip() and cand.strip():
                references.append(ref)
                candidates.append(cand)
                valid_indices.append(i)
        
        if not references:
            return [0.0] * len(text_pairs)
        
        # Force GPU usage if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Batch process ALL texts at once (huge speedup!)
        P, R, F1 = score(candidates, references, 
                         model_type="distilbert-base-uncased", 
                         device=device,
                         verbose=False,
                         batch_size=32)  # Large batch for efficiency
        
        # Map results back to original order
        results = [0.0] * len(text_pairs)
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = F1[i].item()
        
        return results
        
    except Exception as e:
        print(f"Error calculating batch BERTScore: {e}")
        return [0.0] * len(text_pairs)

def calculate_bertscore_fast(reference_text, candidate_text):
    """Single BERTScore calculation (fallback for compatibility)."""
    results = calculate_bertscore_batch([(reference_text, candidate_text)])
    return results[0] if results else 0.0

def calculate_all_scores_cluster_cached(expert_json_str, ai_json_str):
    """Calculate scores using ONLY cached models - no downloads allowed."""
    scores = {}
    
    try:
        # Standard scores (fast, no external dependencies)
        scores['bleu'] = textgrad_train.calculate_bleu_score(expert_json_str, ai_json_str)
        scores['rouge_1'] = textgrad_train.calculate_rouge_score(expert_json_str, ai_json_str)
        scores['exact_match'] = textgrad_train.calculate_exact_field_match(expert_json_str, ai_json_str)
        scores['jaccard'] = textgrad_train.calculate_jaccard_similarity(expert_json_str, ai_json_str)
        
        # METEOR (force cached WordNet only)
        try:
            print("🔄 Calculating METEOR (cached)...")
            scores['meteor'] = textgrad_train.calculate_meteor_score(expert_json_str, ai_json_str)
            print("✅ METEOR done")
        except Exception as e:
            print(f"⚠️  METEOR failed: {e}")
            scores['meteor'] = 0.0
        
        # BERTScore (GPU-optimized version)
        try:
            print("🔄 Calculating BERTScore (GPU-optimized)...")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Device: {device}")
            
            # Use our optimized BERTScore function
            scores['bertscore'] = calculate_bertscore_fast(expert_json_str, ai_json_str)
            print("✅ BERTScore done")
        except Exception as e:
            print(f"⚠️  BERTScore failed: {e}")
            scores['bertscore'] = 0.0
        
    except Exception as e:
        print(f"❌ Error in cluster scoring: {e}")
        # Return zeros for all metrics
        for metric in ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard']:
            scores[metric] = 0.0
    
    return scores

def get_test_files(test_path="split_datasets/fold_3/test"):
    """Get list of JSON files from test set."""
    try:
        test_dir = Path(test_path)
        if not test_dir.exists():
            print(f"Test directory not found: {test_path}")
            return []
        
        json_files = list(test_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in test set")
        return json_files
    except Exception as e:
        print(f"Error accessing test files: {e}")
        return []

def load_checkpoint(output_name="test_eval"):
    """Load checkpoint file to see what's already been processed."""
    checkpoint_path = Path(f"output/{output_name}/checkpoint.json")
    
    if not checkpoint_path.exists():
        print("📍 No checkpoint found - starting fresh")
        return {
            'completed_dois': [],
            'processed_files': {},
            'total_completed': 0
        }
    
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        completed_count = len(checkpoint.get('completed_dois', []))
        print(f"📍 Checkpoint loaded - {completed_count} DOIs already completed")
        return checkpoint
    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e} - starting fresh")
        return {
            'completed_dois': [],
            'processed_files': {},
            'total_completed': 0
        }

def save_checkpoint(checkpoint, output_name="test_eval"):
    """Save current progress to checkpoint file."""
    checkpoint_dir = Path(f"output/{output_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Error saving checkpoint: {e}")

def verify_completion_status(doi, output_name, prompt_versions=["original", "optimized"]):
    """Verify if a DOI has been completely processed for both prompt versions."""
    sanitized_doi = sanitize_filename(doi)
    completion_status = {}
    
    for version in prompt_versions:
        # Check if both output and expert files exist and are valid
        output_path = Path(f"output/{output_name}/{version}/{sanitized_doi}/JSON_output.md")
        expert_path = Path(f"output/{output_name}/{version}/{sanitized_doi}/JSON_expert.md")
        
        output_exists = output_path.exists() and output_path.stat().st_size > 0
        expert_exists = expert_path.exists() and expert_path.stat().st_size > 0
        
        # Additional validation - check if files contain valid content
        valid_output = False
        valid_expert = False
        
        if output_exists:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    valid_output = len(content) > 10  # Basic sanity check
            except:
                valid_output = False
        
        if expert_exists:
            try:
                with open(expert_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    valid_expert = len(content) > 10 and 'records' in content  # Basic sanity check
            except:
                valid_expert = False
        
        completion_status[version] = {
            'output_exists': output_exists,
            'expert_exists': expert_exists,
            'valid_output': valid_output,
            'valid_expert': valid_expert,
            'complete': valid_output and valid_expert
        }
    
    # DOI is fully complete only if both versions are complete
    fully_complete = all(completion_status[v]['complete'] for v in prompt_versions)
    
    return completion_status, fully_complete

def analyze_existing_progress(test_files, output_name="test_eval"):
    """Analyze what's already been processed and what needs to be done."""
    print(f"\n🔍 ANALYZING EXISTING PROGRESS...")
    
    # Load checkpoint
    checkpoint = load_checkpoint(output_name)
    completed_dois_from_checkpoint = set(checkpoint.get('completed_dois', []))
    
    # Verify actual file system state
    verified_completed = []
    needs_processing = []
    partial_completions = []
    
    for test_file in test_files:
        try:
            # Get DOI from file
            data = read_json_file(test_file)
            if not data or 'DOI' not in data:
                print(f"⚠️  Skipping {test_file.name} - no valid DOI found")
                continue
            
            doi = data['DOI']
            
            # Check completion status
            completion_status, fully_complete = verify_completion_status(doi, output_name)
            
            if fully_complete:
                verified_completed.append({
                    'doi': doi,
                    'file': test_file,
                    'status': 'complete'
                })
            else:
                # Check if partially complete
                original_complete = completion_status.get('original', {}).get('complete', False)
                optimized_complete = completion_status.get('optimized', {}).get('complete', False)
                
                if original_complete or optimized_complete:
                    partial_completions.append({
                        'doi': doi,
                        'file': test_file,
                        'original_complete': original_complete,
                        'optimized_complete': optimized_complete,
                        'status': 'partial'
                    })
                else:
                    needs_processing.append({
                        'doi': doi,
                        'file': test_file,
                        'status': 'pending'
                    })
        
        except Exception as e:
            print(f"⚠️  Error analyzing {test_file.name}: {e}")
            continue
    
    # Update checkpoint with verified data
    updated_checkpoint = {
        'completed_dois': [item['doi'] for item in verified_completed],
        'processed_files': {item['doi']: str(item['file']) for item in verified_completed},
        'total_completed': len(verified_completed),
        'last_updated': str(pd.Timestamp.now())
    }
    
    # Report findings
    print(f"\n📊 PROGRESS ANALYSIS RESULTS:")
    print(f"  ✅ Fully completed: {len(verified_completed)} DOIs")
    print(f"  🔄 Partially completed: {len(partial_completions)} DOIs")
    print(f"  ⏳ Needs processing: {len(needs_processing)} DOIs")
    print(f"  📁 Total test files: {len(test_files)} files")
    
    if partial_completions:
        print(f"\n🔄 PARTIAL COMPLETIONS:")
        for item in partial_completions:
            orig_status = "✅" if item['original_complete'] else "❌"
            opt_status = "✅" if item['optimized_complete'] else "❌"
            print(f"  {item['doi']}: Original {orig_status} | Optimized {opt_status}")
    
    if len(verified_completed) > 0:
        print(f"\n⏭️  SKIPPING {len(verified_completed)} already completed DOIs")
    
    remaining_work = needs_processing + partial_completions
    if remaining_work:
        print(f"\n🚀 WILL PROCESS {len(remaining_work)} DOIs")
    else:
        print(f"\n🎉 ALL FILES ALREADY COMPLETED!")
    
    # Save updated checkpoint
    save_checkpoint(updated_checkpoint, output_name)
    
    return {
        'completed': verified_completed,
        'partial': partial_completions,
        'pending': needs_processing,
        'checkpoint': updated_checkpoint
    }

def process_single_test_file(json_file_path, prompt_file_path, output_name="test_eval", prompt_version="original", force_reprocess=False, model="deepseek-r1:14b"):
    """Process a single test file with given prompt, with checkpoint awareness."""
    
    # Read JSON file to get DOI first
    data = read_json_file(json_file_path)
    if not data:
        return None
    
    doi = data.get('DOI', 'unknown_doi')
    
    # Check if already processed (unless forced)
    if not force_reprocess:
        completion_status, fully_complete = verify_completion_status(doi, output_name, [prompt_version])
        if completion_status.get(prompt_version, {}).get('complete', False):
            print(f"⏭️  Skipping {json_file_path.name} ({prompt_version}) - already completed")
            
            # Return existing data for evaluation
            try:
                sanitized_doi = sanitize_filename(doi)
                output_path = Path(f"output/{output_name}/{prompt_version}/{sanitized_doi}/JSON_output.md")
                expert_path = Path(f"output/{output_name}/{prompt_version}/{sanitized_doi}/JSON_expert.md")
                
                with open(output_path, 'r', encoding='utf-8') as f:
                    ai_output_content = f.read()
                
                # Reconstruct expert data
                original_text = data.get('Original_Text', '')
                records = data.get('records', [])
                
                return {
                    'doi': doi,
                    'original_text': original_text,
                    'ai_output': ai_output_content,
                    'expert_output': {"records": records},
                    'prompt_version': prompt_version,
                    'output_file': str(output_path),
                    'expert_file': str(expert_path)
                }
            except Exception as e:
                print(f"⚠️  Error loading existing data for {doi}, will reprocess: {e}")
    
    print(f"🔄 Processing: {json_file_path.name} with {prompt_version} prompt using model {model}")
    
    # Extract required fields
    original_text = data.get('Original_Text', '')
    records = data.get('records', [])
    
    if not original_text:
        print(f"No 'Original_Text' found in {json_file_path}")
        return None
    
    # Save expert data
    if records:
        expert_file = save_expert_data(records, doi, output_name, prompt_version)
        if expert_file:
            print(f"Expert data saved: {expert_file}")
    else:
        print(f"No 'records' found in {json_file_path}")
        return None
    
    # Read prompt
    human_prompt = read_human_prompt(prompt_file_path)
    if not human_prompt:
        return None
    
    # Combine original text and prompt
    combined_prompt = f"{original_text}\n\n{human_prompt}"
    
    print(f"🤖 Sending request to Ollama for DOI: {doi}")
    
    # Call Ollama using fast pip package with specified model
    response = call_ollama_fast(combined_prompt, model)
    if not response:
        return None
    
    # Save response
    output_file = save_response(response, doi, output_name, prompt_version)
    if not output_file:
        return None
    
    # Read the saved AI output for evaluation
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            ai_output_content = f.read()
    except Exception as e:
        print(f"Error reading AI output file: {e}")
        ai_output_content = response  # Fallback to raw response
    
    print(f"✅ Successfully processed {json_file_path.name} -> {output_file}")
    
    # Return data for evaluation
    return {
        'doi': doi,
        'original_text': original_text,
        'ai_output': ai_output_content,
        'expert_output': {"records": records},
        'prompt_version': prompt_version,
        'output_file': output_file,
        'expert_file': expert_file
    }

def calculate_scores_for_doi_fast(expert_json_str, ai_json_str, doi, prompt_version, cached_bertscore):
    """Calculate all scores for a single DOI using cached BERTScore (much faster)."""
    try:
        # Calculate non-BERTScore metrics
        scores = {}
        scores['bleu'] = textgrad_train.calculate_bleu_score(expert_json_str, ai_json_str)
        scores['rouge_1'] = textgrad_train.calculate_rouge_score(expert_json_str, ai_json_str)
        scores['exact_match'] = textgrad_train.calculate_exact_field_match(expert_json_str, ai_json_str)
        scores['jaccard'] = textgrad_train.calculate_jaccard_similarity(expert_json_str, ai_json_str)
        
        # METEOR (fast)
        try:
            scores['meteor'] = textgrad_train.calculate_meteor_score(expert_json_str, ai_json_str)
        except Exception as e:
            scores['meteor'] = 0.0
        
        # Use cached BERTScore (instant!)
        scores['bertscore'] = cached_bertscore
        
        # Calculate committee score
        if scores.get('bertscore', 0) > 0:
            committee_metrics = ['rouge_1', 'meteor', 'bertscore']
            valid_scores = [scores[metric] for metric in committee_metrics if scores.get(metric, 0) > 0]
            committee_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            committee_desc = "ROUGE+METEOR+BERTScore"
        else:
            committee_metrics = ['rouge_1', 'meteor', 'exact_match']
            valid_scores = [scores[metric] for metric in committee_metrics if scores.get(metric, 0) > 0]
            committee_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            committee_desc = "ROUGE+METEOR+ExactMatch"
        
        scores['committee'] = committee_score
        
        print(f"  {prompt_version:12} - Committee: {committee_score:.4f} ({committee_desc})")
        print(f"                 BLEU: {scores['bleu']:.3f} | ROUGE: {scores['rouge_1']:.3f} | METEOR: {scores['meteor']:.3f}")
        print(f"                 BERTScore: {scores['bertscore']:.3f} | ExactMatch: {scores['exact_match']:.3f} | Jaccard: {scores['jaccard']:.3f}")
        
        return scores
    except Exception as e:
        print(f"  {prompt_version:12} - Error calculating scores: {e}")
        return {metric: 0.0 for metric in ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']}

def calculate_scores_for_doi(expert_json_str, ai_json_str, doi, prompt_version):
    """Calculate all scores for a single DOI (cluster-safe version)."""
    try:
        # Calculate scores using cached models
        all_scores = calculate_all_scores_cluster_cached(expert_json_str, ai_json_str)
        
        # Calculate committee score manually (more reliable than original function)
        if all_scores.get('bertscore', 0) > 0:
            # Full committee score: average of ROUGE, METEOR, BERTScore
            committee_metrics = ['rouge_1', 'meteor', 'bertscore']
            valid_scores = [all_scores[metric] for metric in committee_metrics if all_scores.get(metric, 0) > 0]
            committee_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            committee_desc = "ROUGE+METEOR+BERTScore"
        else:
            # Fallback committee score without BERTScore
            committee_metrics = ['rouge_1', 'meteor', 'exact_match']
            valid_scores = [all_scores[metric] for metric in committee_metrics if all_scores.get(metric, 0) > 0]
            committee_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            committee_desc = "ROUGE+METEOR+ExactMatch"
        
        all_scores['committee'] = committee_score
        
        print(f"  {prompt_version:12} - Committee: {committee_score:.4f} ({committee_desc})")
        print(f"                 BLEU: {all_scores['bleu']:.3f} | ROUGE: {all_scores['rouge_1']:.3f} | METEOR: {all_scores['meteor']:.3f}")
        print(f"                 BERTScore: {all_scores['bertscore']:.3f} | ExactMatch: {all_scores['exact_match']:.3f} | Jaccard: {all_scores['jaccard']:.3f}")
        
        return all_scores
    except Exception as e:
        print(f"  {prompt_version:12} - Error calculating scores: {e}")
        return {metric: 0.0 for metric in ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']}

def evaluate_test_set(test_files, original_prompt_path, optimized_prompt_path, output_name="test_eval", 
                     primary_metric="bertscore", committee_metrics="committee_default", force_reprocess=False, model="deepseek-r1:14b"):
    """Evaluate test set with both original and optimized prompts with robust resumption."""
    
    print(f"\n{'='*60}")
    print("🧪 TEST SET EVALUATION WITH CHECKPOINT RESUMPTION")
    print(f"Original Prompt: {original_prompt_path}")
    print(f"Optimized Prompt: {optimized_prompt_path}")
    print(f"Test Files: {len(test_files)}")
    print(f"Model: {model}")
    print(f"Force Reprocess: {force_reprocess}")
    print(f"{'='*60}")
    
    # Analyze existing progress
    progress = analyze_existing_progress(test_files, output_name)
    
    # Determine what needs to be processed
    files_to_process = []
    
    # Add pending files (need both original and optimized)
    for item in progress['pending']:
        files_to_process.append({
            'file': item['file'],
            'doi': item['doi'],
            'needs_original': True,
            'needs_optimized': True
        })
    
    # Add partial files (need missing prompt version)
    for item in progress['partial']:
        files_to_process.append({
            'file': item['file'],
            'doi': item['doi'],
            'needs_original': not item['original_complete'],
            'needs_optimized': not item['optimized_complete']
        })
    
    if not files_to_process and not force_reprocess:
        print(f"\n🎉 ALL FILES ALREADY COMPLETED! Loading for batch scoring...")
        # Load all completed results for batch scoring
        results = {}
        for item in progress['completed']:
            try:
                # Load existing results for batch scoring
                doi = item['doi']
                test_file = item['file']
                
                # Get file data for expert output reconstruction
                data = read_json_file(test_file)
                if not data:
                    continue
                
                # Load both output files
                sanitized_doi = sanitize_filename(doi)  # FIXED: Proper indentation
                original_path = Path(f"output/{output_name}/original/{sanitized_doi}/JSON_output.md")
                optimized_path = Path(f"output/{output_name}/optimized/{sanitized_doi}/JSON_output.md")
                    
                if original_path.exists() and optimized_path.exists():
                    with open(original_path, 'r', encoding='utf-8') as f:
                        original_output = f.read()
                    with open(optimized_path, 'r', encoding='utf-8') as f:
                        optimized_output = f.read()
                    
                    # Store for batch scoring (same format as new processing)
                    results[doi] = {
                        'original_output': original_output,
                        'optimized_output': optimized_output,
                        'expert_output': {"records": data.get('records', [])}
                    }
                
            except Exception as e:
                print(f"⚠️  Error loading completed result for {item['doi']}: {e}")
                continue
        
        processed_count = len(progress['completed'])
    else:
        # Process remaining files
        print(f"\n🔄 PROCESSING {len(files_to_process)} FILES...")
        results = {}
        processed_count = 0
        checkpoint = progress['checkpoint']
        
        for i, file_info in enumerate(files_to_process, 1):
            test_file = file_info['file']
            doi = file_info['doi']
            
            try:
                print(f"\n📄 [{i}/{len(files_to_process)}] Processing {test_file.name}")
                print(f"    DOI: {doi}")
                print(f"    Needs: Original={file_info['needs_original']}, Optimized={file_info['needs_optimized']}")
                
                # Process with original prompt if needed
                original_result = None
                if file_info['needs_original'] or force_reprocess:
                    original_result = process_single_test_file(
                        test_file, original_prompt_path, output_name, "original", force_reprocess, model
                    )
                else:
                    # Load existing result
                    original_result = process_single_test_file(
                        test_file, original_prompt_path, output_name, "original", False, model
                    )
                
                if not original_result:
                    print(f"❌ Failed to process {test_file.name} with original prompt")
                    continue
                
                # Process with optimized prompt if needed
                optimized_result = None
                if file_info['needs_optimized'] or force_reprocess:
                    optimized_result = process_single_test_file(
                        test_file, optimized_prompt_path, output_name, "optimized", force_reprocess, model
                    )
                else:
                    # Load existing result
                    optimized_result = process_single_test_file(
                        test_file, optimized_prompt_path, output_name, "optimized", False, model
                    )
                
                if not optimized_result:
                    print(f"❌ Failed to process {test_file.name} with optimized prompt")
                    continue
                
                # Store results for later batch scoring (much faster!)
                results[doi] = {
                    'original_output': original_result['ai_output'],
                    'optimized_output': optimized_result['ai_output'], 
                    'expert_output': original_result['expert_output']
                }
                
                print(f"✅ Data stored for batch scoring (skipping individual scoring for speed)")
                
                processed_count += 1
                
                # Update checkpoint after each successful processing
                if doi not in checkpoint['completed_dois']:
                    checkpoint['completed_dois'].append(doi)
                    checkpoint['processed_files'][doi] = str(test_file)
                    checkpoint['total_completed'] = len(checkpoint['completed_dois'])
                    checkpoint['last_updated'] = str(pd.Timestamp.now())
                    save_checkpoint(checkpoint, output_name)
                
                print(f"✅ Completed {processed_count}/{len(files_to_process)} files")
                
            except Exception as e:
                print(f"❌ Error processing {test_file.name}: {e}")
                continue
    
    # Add results from already completed files (for batch scoring)
    for item in progress['completed']:
        if item['doi'] not in results:
            # Load these for batch scoring if not already loaded above
            try:
                doi = item['doi']
                test_file = item['file']
                
                # Get file data for expert output reconstruction
                data = read_json_file(test_file)
                if not data:
                    continue
                
                # Load both output files
                sanitized_doi = sanitize_filename(doi)  # FIXED: Proper indentation
                original_path = Path(f"output/{output_name}/original/{sanitized_doi}/JSON_output.md")
                optimized_path = Path(f"output/{output_name}/optimized/{sanitized_doi}/JSON_output.md")
                    
                if original_path.exists() and optimized_path.exists():
                    with open(original_path, 'r', encoding='utf-8') as f:
                        original_output = f.read()
                    with open(optimized_path, 'r', encoding='utf-8') as f:
                        optimized_output = f.read()
                    
                    # Store for batch scoring (same format as new processing)
                    results[doi] = {
                        'original_output': original_output,
                        'optimized_output': optimized_output,
                        'expert_output': {"records": data.get('records', [])}
                    }
                
            except Exception as e:
                print(f"⚠️  Error loading completed result for {item['doi']}: {e}")
                continue
    
    total_completed = len(progress['completed']) + processed_count
    
    if not results:
        print("❌ No files were successfully processed")
        return None
    
    # BATCH SCORING PHASE - Much faster than individual scoring!
    print(f"\n{'='*60}")
    print("📊 BATCH SCORING PHASE - CALCULATING ALL METRICS")
    print(f"Processing {len(results)} DOIs with optimized batch scoring...")
    print(f"{'='*60}")
    
    # Check for existing scoring checkpoint
    scoring_checkpoint_path = Path(f"output/{output_name}/scoring_checkpoint.json")
    scored_results = {}
    
    if scoring_checkpoint_path.exists():
        try:
            with open(scoring_checkpoint_path, 'r', encoding='utf-8') as f:
                scored_results = json.load(f)
            print(f"📍 Scoring checkpoint loaded - {len(scored_results)} DOIs already scored")
        except Exception as e:
            print(f"⚠️  Error loading scoring checkpoint: {e}")
            scored_results = {}
    
    # Also check for partial results from previous wall time interruption
    partial_results_path = Path("test_evaluation_results") / f"{output_name}_partial_results.json"
    if partial_results_path.exists() and len(scored_results) == 0:
        try:
            with open(partial_results_path, 'r', encoding='utf-8') as f:
                partial_data = json.load(f)
            scored_results = partial_data.get('scored_results', {})
            metadata = partial_data.get('metadata', {})
            print(f"🔄 Recovered partial results from previous run - {len(scored_results)} DOIs")
            print(f"   Previous completion: {metadata.get('completion_percentage', 0):.1f}%")
        except Exception as e:
            print(f"⚠️  Error loading partial results: {e}")
    
    # Convert stored data to scored results (skip already scored)
    total_dois = len(results)
    remaining_dois = [doi for doi in results.keys() if doi not in scored_results]
    
    print(f"✅ Already scored: {len(scored_results)} DOIs")
    print(f"⏳ Need to score: {len(remaining_dois)} DOIs")
    
    # BATCH BERTSCORE OPTIMIZATION - Process all BERTScores at once!
    if remaining_dois:
        print(f"\n🚀 BATCH BERTSCORE OPTIMIZATION - Processing {len(remaining_dois)} DOIs")
        
        # Prepare all text pairs for batch BERTScore
        bertscore_pairs = []
        doi_mapping = []
        
        for doi in remaining_dois:
            stored_data = results[doi]
            expert_json_str = extract_json_from_content(json.dumps(stored_data['expert_output']))
            original_ai_json = extract_json_from_content(stored_data['original_output'])
            optimized_ai_json = extract_json_from_content(stored_data['optimized_output'])
            
            if expert_json_str and original_ai_json:
                bertscore_pairs.append((expert_json_str, original_ai_json))
                doi_mapping.append((doi, 'original'))
            
            if expert_json_str and optimized_ai_json:
                bertscore_pairs.append((expert_json_str, optimized_ai_json))
                doi_mapping.append((doi, 'optimized'))
        
        # Calculate ALL BERTScores at once (huge speedup!)
        print(f"🔄 Calculating {len(bertscore_pairs)} BERTScores in one batch...")
        batch_bertscores = calculate_bertscore_batch(bertscore_pairs)
        print(f"✅ Batch BERTScore completed in seconds instead of hours!")
        
        # Store BERTScore results
        bertscore_cache = {}
        for i, (doi, prompt_type) in enumerate(doi_mapping):
            if doi not in bertscore_cache:
                bertscore_cache[doi] = {}
            bertscore_cache[doi][prompt_type] = batch_bertscores[i] if i < len(batch_bertscores) else 0.0
    
    # Now process each DOI with cached BERTScores
    for i, doi in enumerate(remaining_dois, 1):
        stored_data = results[doi]
        print(f"\n📊 [{len(scored_results)+i}/{total_dois}] Scoring DOI: {doi}")
        
        try:
            # Get expert JSON
            expert_json_str = extract_json_from_content(json.dumps(stored_data['expert_output']))
            if not expert_json_str:
                print(f"❌ Could not extract expert JSON for {doi}")
                continue
            
            scored_results[doi] = {}
            
            # Score original prompt (using cached BERTScore)
            original_ai_json = extract_json_from_content(stored_data['original_output'])
            if original_ai_json:
                original_scores = calculate_scores_for_doi_fast(expert_json_str, original_ai_json, doi, "original", bertscore_cache.get(doi, {}).get('original', 0.0))
                scored_results[doi]['original'] = original_scores
            else:
                print(f"  original     - Could not extract AI JSON")
                scored_results[doi]['original'] = {metric: 0.0 for metric in ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']}
            
            # Score optimized prompt (using cached BERTScore)
            optimized_ai_json = extract_json_from_content(stored_data['optimized_output'])
            if optimized_ai_json:
                optimized_scores = calculate_scores_for_doi_fast(expert_json_str, optimized_ai_json, doi, "optimized", bertscore_cache.get(doi, {}).get('optimized', 0.0))
                scored_results[doi]['optimized'] = optimized_scores
            else:
                print(f"  optimized    - Could not extract AI JSON")
                scored_results[doi]['optimized'] = {metric: 0.0 for metric in ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']}
            
            # Save scoring checkpoint AND partial results every 10 DOIs (faster now)
            if i % 10 == 0:
                try:
                    with open(scoring_checkpoint_path, 'w', encoding='utf-8') as f:
                        json.dump(scored_results, f, indent=2, ensure_ascii=False)
                    print(f"💾 Scoring checkpoint saved ({len(scored_results)} DOIs)")
                    
                    # Also save partial results for wall time protection
                    save_partial_results(scored_results, output_name, len(scored_results), total_dois)
                    
                except Exception as e:
                    print(f"⚠️  Error saving scoring checkpoint: {e}")
                
        except Exception as e:
            print(f"❌ Error scoring {doi}: {e}")
            continue
    
    # Final scoring checkpoint save
    try:
        with open(scoring_checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(scored_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Final scoring checkpoint saved ({len(scored_results)} DOIs)")
        
        # Save final partial results (in case final analysis fails)
        save_partial_results(scored_results, output_name, len(scored_results), total_dois)
        
    except Exception as e:
        print(f"⚠️  Error saving final scoring checkpoint: {e}")
    
    # Use scored results for final analysis
    results = scored_results
    
    # Calculate summary statistics
    print(f"\n{'='*60}")
    print("📊 CALCULATING SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    metrics = ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']
    summary_stats = {}
    
    for prompt_type in ['original', 'optimized']:
        summary_stats[prompt_type] = {}
        
        for metric in metrics:
            scores_list = []
            for doi in results:
                if prompt_type in results[doi] and metric in results[doi][prompt_type]:
                    scores_list.append(results[doi][prompt_type][metric])
            
            if scores_list:
                summary_stats[prompt_type][metric] = {
                    'mean': sum(scores_list) / len(scores_list),
                    'std': (sum((x - sum(scores_list)/len(scores_list))**2 for x in scores_list) / len(scores_list))**0.5 if len(scores_list) > 1 else 0.0,
                    'min': min(scores_list),
                    'max': max(scores_list),
                    'count': len(scores_list)
                }
            else:
                summary_stats[prompt_type][metric] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                }
    
    # Display results
    print(f"\n🔸 ORIGINAL PROMPT PERFORMANCE:")
    for metric in metrics:
        stats = summary_stats['original'][metric]
        print(f"  {metric.upper():12}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.3f}, max: {stats['max']:.3f})")
    
    print(f"\n🔸 OPTIMIZED PROMPT PERFORMANCE:")
    for metric in metrics:
        stats = summary_stats['optimized'][metric]
        print(f"  {metric.upper():12}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.3f}, max: {stats['max']:.3f})")
    
    # Calculate improvements
    print(f"\n🔸 IMPROVEMENT ANALYSIS:")
    improvements = {}
    for metric in metrics:
        original_mean = summary_stats['original'][metric]['mean']
        optimized_mean = summary_stats['optimized'][metric]['mean']
        
        if original_mean > 0:
            improvement_pct = ((optimized_mean - original_mean) / original_mean) * 100
            improvement_abs = optimized_mean - original_mean
            improvements[metric] = {
                'absolute': improvement_abs,
                'percentage': improvement_pct,
                'better': optimized_mean > original_mean
            }
            
            status = "🟢 IMPROVED" if improvement_pct > 0 else "🔴 DECLINED" if improvement_pct < 0 else "⚪ UNCHANGED"
            print(f"  {metric.upper():12}: {improvement_abs:+.4f} ({improvement_pct:+.1f}%) {status}")
        else:
            improvements[metric] = {'absolute': 0.0, 'percentage': 0.0, 'better': False}
            print(f"  {metric.upper():12}: N/A (original score was 0)")
    
    # Overall winner
    primary_improvement = improvements.get(primary_metric, {})
    committee_improvement = improvements.get('committee', {})
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"  Primary Metric ({primary_metric.upper()}): {'OPTIMIZED WINS' if primary_improvement.get('better', False) else 'ORIGINAL WINS'}")
    print(f"  Committee Score: {'OPTIMIZED WINS' if committee_improvement.get('better', False) else 'ORIGINAL WINS'}")
    
    # Count wins per metric
    optimized_wins = sum(1 for metric in improvements if improvements[metric].get('better', False))
    total_metrics = len(metrics)
    
    print(f"  Metric Wins: OPTIMIZED {optimized_wins}/{total_metrics}, ORIGINAL {total_metrics - optimized_wins}/{total_metrics}")
    
    overall_winner = "OPTIMIZED" if optimized_wins > total_metrics // 2 else "ORIGINAL"
    print(f"  🎯 OVERALL WINNER: {overall_winner} PROMPT")
    
    # Save results
    save_test_evaluation_results(results, summary_stats, improvements, output_name, 
                                original_prompt_path, optimized_prompt_path, total_completed)
    
    return {
        'detailed_results': results,
        'summary_stats': summary_stats,
        'improvements': improvements,
        'overall_winner': overall_winner,
        'processed_count': total_completed
    }

def save_partial_results(scored_results, output_name, completed_count, total_count):
    """Save partial results during scoring to prevent loss from wall time."""
    
    # Create output directory
    output_dir = Path("test_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save partial detailed results JSON
    partial_filepath = output_dir / f"{output_name}_partial_results.json"
    partial_data = {
        'metadata': {
            'output_name': output_name,
            'completed_count': completed_count,
            'total_count': total_count,
            'completion_percentage': (completed_count / total_count * 100) if total_count > 0 else 0,
            'timestamp': str(pd.Timestamp.now()),
            'status': 'partial_scoring_in_progress'
        },
        'scored_results': scored_results
    }
    
    try:
        with open(partial_filepath, 'w', encoding='utf-8') as f:
            json.dump(partial_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Partial results saved: {completed_count}/{total_count} DOIs ({completed_count/total_count*100:.1f}%)")
    except Exception as e:
        print(f"⚠️  Error saving partial results: {e}")

def save_test_evaluation_results(results, summary_stats, improvements, output_name, 
                               original_prompt_path, optimized_prompt_path, processed_count):
    """Save comprehensive test evaluation results."""
    
    # Create output directory
    output_dir = Path("test_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results JSON
    detailed_filepath = output_dir / f"{output_name}_detailed_results.json"
    detailed_data = {
        'metadata': {
            'output_name': output_name,
            'original_prompt_path': original_prompt_path,
            'optimized_prompt_path': optimized_prompt_path,
            'processed_count': processed_count,
            'total_dois': len(results)
        },
        'detailed_results': results,
        'summary_statistics': summary_stats,
        'improvements': improvements
    }
    
    try:
        with open(detailed_filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to: {detailed_filepath}")
    except Exception as e:
        print(f"Error saving detailed results: {e}")
    
    # Save summary report
    create_test_summary_report(summary_stats, improvements, output_dir, output_name, 
                              original_prompt_path, optimized_prompt_path, processed_count)
    
    # Save CSV for analysis
    create_test_results_csv(results, output_dir, output_name)

def create_test_summary_report(summary_stats, improvements, output_dir, output_name,
                              original_prompt_path, optimized_prompt_path, processed_count):
    """Create a markdown summary report."""
    
    # Determine overall winner
    optimized_wins = sum(1 for metric in improvements if improvements[metric].get('better', False))
    total_metrics = len(improvements)
    overall_winner = "OPTIMIZED" if optimized_wins > total_metrics // 2 else "ORIGINAL"
    
    report_content = f"""# 🧪 Test Set Evaluation Report

## 📋 Configuration
- **Output Name**: {output_name}
- **Original Prompt**: `{original_prompt_path}`
- **Optimized Prompt**: `{optimized_prompt_path}`
- **Test Papers Processed**: {processed_count}

## 📊 Performance Summary

### Original Prompt Performance
| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
"""
    
    metrics = ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']
    
    for metric in metrics:
        stats = summary_stats['original'][metric]
        report_content += f"| {metric.upper()} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.3f} | {stats['max']:.3f} |\n"
    
    report_content += f"""
### Optimized Prompt Performance
| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
"""
    
    for metric in metrics:
        stats = summary_stats['optimized'][metric]
        report_content += f"| {metric.upper()} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.3f} | {stats['max']:.3f} |\n"
    
    report_content += f"""
## 🔄 Improvement Analysis

| Metric | Absolute Δ | Relative Δ | Status |
|--------|------------|------------|---------|
"""
    
    for metric in metrics:
        improvement = improvements[metric]
        status = "🟢 IMPROVED" if improvement['percentage'] > 0 else "🔴 DECLINED" if improvement['percentage'] < 0 else "⚪ UNCHANGED"
        report_content += f"| {metric.upper()} | {improvement['absolute']:+.4f} | {improvement['percentage']:+.1f}% | {status} |\n"
    
    report_content += f"""
## 🏆 Final Results

### Winner Distribution
- **🚀 Optimized Prompt Wins**: {optimized_wins}/{total_metrics} metrics ({optimized_wins/total_metrics*100:.1f}%)
- **🏠 Original Prompt Wins**: {total_metrics - optimized_wins}/{total_metrics} metrics ({(total_metrics - optimized_wins)/total_metrics*100:.1f}%)

### 🎯 Overall Winner: **{overall_winner} PROMPT**

## 📈 Key Insights

### Best Performing Metrics (Optimized)
"""
    
    # Sort improvements by percentage
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1]['percentage'], reverse=True)
    
    for metric, improvement in sorted_improvements[:3]:
        if improvement['percentage'] > 0:
            report_content += f"- **{metric.upper()}**: +{improvement['percentage']:.1f}% improvement\n"
    
    report_content += f"""
### Areas for Further Improvement
"""
    
    for metric, improvement in sorted_improvements[-3:]:
        if improvement['percentage'] < 0:
            report_content += f"- **{metric.upper()}**: {improvement['percentage']:.1f}% decline\n"
    
    # Count API usage
    report_content += f"""
## 🔧 Technical Details

### API Usage
- **Total Ollama Calls**: {API_USAGE_TRACKER['ollama_calls']:,}
- **Total Response Tokens**: {API_USAGE_TRACKER['ollama_response_tokens']:,}
- **Average Tokens per Call**: {API_USAGE_TRACKER['ollama_response_tokens'] // max(1, API_USAGE_TRACKER['ollama_calls']):,}

### Evaluation Methodology
- **Metrics Used**: BLEU, ROUGE-1, METEOR, BERTScore, Exact Match, Jaccard, Committee Score
- **Committee Score**: Average of BERTScore, ROUGE-1, and METEOR (most trusted metrics)
- **Processing**: Each test paper processed with both original and optimized prompts
- **Comparison**: Direct head-to-head comparison on identical test data

---
*Generated automatically by Test Set Evaluation System*
"""
    
    # Save the report
    try:
        report_filepath = output_dir / f"{output_name}_summary_report.md"
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📊 Summary report saved to: {report_filepath}")
    except Exception as e:
        print(f"Error saving summary report: {e}")

def create_test_results_csv(results, output_dir, output_name):
    """Create CSV file for detailed analysis."""
    
    try:
        # Prepare data for CSV
        csv_data = []
        metrics = ['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee']
        
        for doi, doi_results in results.items():
            row = {'DOI': doi}
            
            # Add original scores
            for metric in metrics:
                original_score = doi_results.get('original', {}).get(metric, 0.0)
                row[f'original_{metric}'] = original_score
            
            # Add optimized scores
            for metric in metrics:
                optimized_score = doi_results.get('optimized', {}).get(metric, 0.0)
                row[f'optimized_{metric}'] = optimized_score
            
            # Add improvements
            for metric in metrics:
                original_score = doi_results.get('original', {}).get(metric, 0.0)
                optimized_score = doi_results.get('optimized', {}).get(metric, 0.0)
                improvement = optimized_score - original_score
                row[f'improvement_{metric}'] = improvement
                
                if original_score > 0:
                    improvement_pct = (improvement / original_score) * 100
                    row[f'improvement_pct_{metric}'] = improvement_pct
                else:
                    row[f'improvement_pct_{metric}'] = 0.0
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        csv_filepath = output_dir / f"{output_name}_detailed_results.csv"
        df.to_csv(csv_filepath, index=False)
        print(f"📊 CSV results saved to: {csv_filepath}")
        
    except Exception as e:
        print(f"Error creating CSV: {e}")

def main():
    """Main function to handle command line arguments and run test evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate prompt performance on test set')
    parser.add_argument('--test-path', type=str, default='split_datasets/fold_3/test',
                       help='Path to test set directory (default: split_datasets/fold_3/test)')
    parser.add_argument('--original-prompt', type=str, default='human_init_prompt.md',
                       help='Path to original human prompt (default: human_init_prompt.md)')
    parser.add_argument('--optimized-prompt', type=str, default='final_results/final_optimized_prompt.md',
                       help='Path to optimized prompt (default: final_results/final_optimized_prompt.md)')
    parser.add_argument('--output-name', type=str, default='test_eval',
                       help='Output name for results (default: test_eval)')
    parser.add_argument('--primary-metric', type=str, default='bertscore',
                       choices=['bleu', 'rouge_1', 'meteor', 'bertscore', 'exact_match', 'jaccard', 'committee'],
                       help='Primary metric for evaluation (default: bertscore)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of test files to process (default: all)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of all files, ignoring existing results (default: False)')
    parser.add_argument('--model', type=str, default='deepseek-r1:14b',
                       help='Ollama model to use for processing (default: deepseek-r1:14b)')
    
    args = parser.parse_args()
    
    # Initialize real-time logging
    initialize_logging(args.output_name)
    
    # Check if prompt files exist
    if not os.path.exists(args.original_prompt):
        print(f"Error: Original prompt file '{args.original_prompt}' not found.")
        sys.exit(1)
    
    if not os.path.exists(args.optimized_prompt):
        print(f"Error: Optimized prompt file '{args.optimized_prompt}' not found.")
        sys.exit(1)
    
    # Get test files
    test_files = get_test_files(args.test_path)
    if not test_files:
        print("No test files found!")
        sys.exit(1)
    
    # Limit files if specified
    if args.max_files:
        test_files = test_files[:args.max_files]
        print(f"Processing limited to {len(test_files)} files")
    
    print(f"\n🧪 TEST SET EVALUATION CONFIGURATION:")
    print(f"  Test Set Path: {args.test_path}")
    print(f"  Original Prompt: {args.original_prompt}")
    print(f"  Optimized Prompt: {args.optimized_prompt}")
    print(f"  Output Name: {args.output_name}")
    print(f"  Primary Metric: {args.primary_metric}")
    print(f"  Model: {args.model}")
    print(f"  Test Files: {len(test_files)}")
    print(f"  Force Reprocess: {args.force_reprocess}")
    
    # Run evaluation
    results = evaluate_test_set(
        test_files=test_files,
        original_prompt_path=args.original_prompt,
        optimized_prompt_path=args.optimized_prompt,
        output_name=args.output_name,
        primary_metric=args.primary_metric,
        force_reprocess=args.force_reprocess,
        model=args.model
    )
    
    if results:
        print(f"\n🎉 TEST SET EVALUATION COMPLETED!")
        print(f"📁 Results saved in: ./test_evaluation_results/")
        print(f"🏆 Overall Winner: {results['overall_winner']} PROMPT")
        print(f"📊 Processed: {results['processed_count']} test papers")
        print(f"🤖 Model Used: {args.model}")
        
        # Print API usage summary
        print(f"\n📊 API USAGE SUMMARY:")
        print(f"   Ollama calls: {API_USAGE_TRACKER['ollama_calls']:,}")
        print(f"   Total tokens: {API_USAGE_TRACKER['ollama_response_tokens']:,}")
    else:
        print("\n❌ Test set evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()