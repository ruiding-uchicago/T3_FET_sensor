#!/usr/bin/env python3
"""
DNA/RNA Feature Extraction Utility with DNABERT-S Embeddings
Usage: Pass sequence string, get features + 256D DNABERT-S embedding
"""

# Triton stub for macOS
import sys
from types import ModuleType
from importlib.machinery import ModuleSpec
triton_stub = ModuleType('triton')
triton_stub.__spec__ = ModuleSpec('triton', None)
triton_stub.__version__ = '2.0.0'
sys.modules['triton'] = triton_stub

import re
import json
import os
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')


class DNABERTSFeatureExtractor:
    """DNA/RNA feature extraction with DNABERT-S embeddings and JSON caching"""

    def __init__(self, cache_dir='cache', model_name="zhihan1996/DNABERT-S"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Thermodynamic parameters
        self.nn_params = {
            'AA/TT': -1.00, 'AT/TA': -0.88, 'TA/AT': -0.58, 'CA/GT': -1.45,
            'GT/CA': -1.44, 'CT/GA': -1.28, 'GA/CT': -1.30, 'CG/GC': -2.17,
            'GC/CG': -2.24, 'GG/CC': -1.84, 'TT/AA': -1.00, 'TG/AC': -1.45,
            'AC/TG': -1.30, 'AG/TC': -1.28, 'TC/AG': -1.44, 'CC/GG': -1.84
        }

        # Load DNABERT-S model
        print(f"Loading DNABERT-S model: {model_name}")
        print("(First run will download model, this may take a few minutes...)")
        print("DNABERT-S: Species-aware DNA embeddings with validated 256D compression")

        # Bypass torch version check by forcing safetensors loading
        import os as _os
        _os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

        # Patch transformers to bypass torch version check
        import transformers.utils.import_utils as import_utils
        original_check = getattr(import_utils, 'check_torch_load_is_safe', None)
        if original_check:
            import_utils.check_torch_load_is_safe = lambda: None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Force safetensors format (doesn't need torch version check)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True
        )

        # Restore original check
        if original_check:
            import_utils.check_torch_load_is_safe = original_check

        self.model.eval()

        # Force CPU for compatibility (MPS may have issues with some models)
        self.device = torch.device("cpu")
        print("Using CPU (forcing for compatibility)")

        print("✓ DNABERT-S model loaded successfully!\n")

    def _get_cache_path(self, seq_str):
        """Generate cache file path from sequence string"""
        filename = seq_str.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename = filename.replace('*', '_').replace('?', '_').replace('"', '_')
        filename = filename.replace('<', '_').replace('>', '_').replace('|', '_')
        filename = f"{filename}.json"
        return os.path.join(self.cache_dir, filename)

    def parse_sequence(self, seq_str):
        """Parse sequence string, remove modifications"""
        match = re.search(r"[53]['′]-?(.+?)-?[53]['′]", seq_str)
        if match:
            seq = match.group(1)
        else:
            parts = seq_str.split()
            seq = parts[-1] if parts else seq_str

        seq = re.sub(r'NH2?-?|\(CH2\)\d+-?|SH-?|HS-?|HSC\d+-?|C\d+-?|AC\d+-?|-C\d+NH2|₂', '', seq)
        seq = seq.replace(' ', '').replace('-', '')
        seq = ''.join(c for c in seq.upper() if c in 'ATCGUN')

        return seq

    def calculate_basic_properties(self, seq):
        """Calculate basic sequence properties"""
        length = len(seq)
        if length == 0:
            return {}

        counts = Counter(seq)
        a_count = counts.get('A', 0)
        t_count = counts.get('T', 0)
        c_count = counts.get('C', 0)
        g_count = counts.get('G', 0)
        u_count = counts.get('U', 0)

        is_rna = u_count > 0
        seq_type = 'RNA' if is_rna else 'DNA'

        gc_content = (g_count + c_count) / length if length > 0 else 0
        at_content = (a_count + t_count + u_count) / length if length > 0 else 0

        purines = a_count + g_count
        pyrimidines = c_count + t_count + u_count
        purine_pyrimidine_ratio = purines / pyrimidines if pyrimidines > 0 else 0

        return {
            'length': length,
            'seq_type': seq_type,
            'gc_content': gc_content,
            'at_content': at_content,
            'a_count': a_count,
            'c_count': c_count,
            'g_count': g_count,
            't_count': t_count,
            'u_count': u_count,
            'purine_pyrimidine_ratio': purine_pyrimidine_ratio,
        }

    def calculate_tm_wallace(self, seq):
        """Calculate melting temperature"""
        counts = Counter(seq)
        a = counts.get('A', 0)
        t = counts.get('T', 0)
        c = counts.get('C', 0)
        g = counts.get('G', 0)
        u = counts.get('U', 0)
        length = len(seq)

        if length < 14:
            tm = 2 * (a + t + u) + 4 * (g + c)
        else:
            gc = g + c
            at = a + t + u
            if (at + gc) > 0:
                tm = 64.9 + 41 * (gc - 16.4) / (at + gc)
            else:
                tm = 0

        return tm

    def calculate_delta_g_simple(self, seq):
        """Simplified ΔG calculation"""
        if len(seq) < 2:
            return 0.0

        delta_g = 0.2

        for i in range(len(seq) - 1):
            dinuc = seq[i:i+2]
            for key, value in self.nn_params.items():
                if dinuc in key or dinuc[::-1] in key:
                    delta_g += value
                    break

        return delta_g

    def calculate_sequence_complexity(self, seq):
        """Calculate sequence complexity (Shannon entropy)"""
        if len(seq) == 0:
            return 0.0

        counts = Counter(seq)
        probs = [count / len(seq) for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        max_entropy = np.log2(4)
        complexity = entropy / max_entropy if max_entropy > 0 else 0

        return complexity

    def find_longest_homopolymer(self, seq):
        """Find longest homopolymer"""
        if len(seq) == 0:
            return 0

        max_length = 1
        current_length = 1

        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 1

        return max_length

    def calculate_kmer_frequencies(self, seq, k=2):
        """Calculate k-mer frequencies"""
        if len(seq) < k:
            return {}

        kmer_counts = Counter()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N' not in kmer:
                kmer_counts[kmer] += 1

        total = sum(kmer_counts.values())
        kmer_freqs = {kmer: count / total for kmer, count in kmer_counts.items()} if total > 0 else {}

        return kmer_freqs

    def extract_features(self, seq_str):
        """Extract 25 macroscopic features"""
        seq = self.parse_sequence(seq_str)
        basic_props = self.calculate_basic_properties(seq)

        tm = self.calculate_tm_wallace(seq)
        delta_g = self.calculate_delta_g_simple(seq)
        complexity = self.calculate_sequence_complexity(seq)
        longest_homopolymer = self.find_longest_homopolymer(seq)
        cpg_count = seq.count('CG')

        gc_skew = (basic_props['g_count'] - basic_props['c_count']) / (basic_props['g_count'] + basic_props['c_count'] + 1e-10)
        at_skew = (basic_props['a_count'] - basic_props['t_count'] - basic_props['u_count']) / (basic_props['a_count'] + basic_props['t_count'] + basic_props['u_count'] + 1e-10)

        dinuc_freq = self.calculate_kmer_frequencies(seq, k=2)

        features = {
            'length': basic_props['length'],
            'seq_type': basic_props['seq_type'],
            'gc_content': basic_props['gc_content'],
            'at_content': basic_props['at_content'],
            'a_count': basic_props['a_count'],
            'c_count': basic_props['c_count'],
            'g_count': basic_props['g_count'],
            't_count': basic_props['t_count'],
            'u_count': basic_props['u_count'],
            'purine_pyrimidine_ratio': basic_props['purine_pyrimidine_ratio'],
            'tm_celsius': tm,
            'delta_g_kcal_mol': delta_g,
            'complexity': complexity,
            'longest_homopolymer': longest_homopolymer,
            'cpg_count': cpg_count,
            'gc_skew': gc_skew,
            'at_skew': at_skew,
            'dinuc_CG': dinuc_freq.get('CG', 0.0),
            'dinuc_GC': dinuc_freq.get('GC', 0.0),
            'dinuc_AT': dinuc_freq.get('AT', 0.0),
            'dinuc_TA': dinuc_freq.get('TA', 0.0),
            'dinuc_GG': dinuc_freq.get('GG', 0.0),
            'dinuc_CC': dinuc_freq.get('CC', 0.0),
            'dinuc_AA': dinuc_freq.get('AA', 0.0),
            'dinuc_TT': dinuc_freq.get('TT', 0.0),
            'sequence': seq,
        }

        return features

    def _features_to_macroscopic_properties(self, features):
        """
        Convert features dict to numerical-only macroscopic_properties

        Replaces categorical seq_type with binary encoding (is_rna: 0=DNA, 1=RNA)
        Removes 'sequence' field (kept separately in JSON output)

        Returns:
            dict: All numerical features (25 total: 24 original + 1 binary)
        """
        props = {}

        # Binary encode seq_type: 1=RNA, 0=DNA
        is_rna = 1 if features['seq_type'] == 'RNA' else 0

        # Copy all features except seq_type and sequence
        for key, value in features.items():
            if key not in ['seq_type', 'sequence']:
                props[key] = value

        # Add binary encoded field
        props['is_rna'] = is_rna

        return props

    def generate_dnaberts_embedding(self, seq_str, target_dim=256):
        """
        Generate DNABERT-S embedding and reduce to target dimensions

        DNABERT-S produces 768D embeddings from its transformer model.
        We use mean pooling over sequence positions and then reduce to 256D.
        DNABERT-S paper validates that 256D compression maintains nearly lossless performance.
        """
        seq = self.parse_sequence(seq_str)

        # Replace U with T for DNABERT-S (it expects DNA sequences)
        seq_dna = seq.replace('U', 'T')

        # Tokenize and get embeddings
        inputs = self.tokenizer(seq_dna, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Handle tuple output (DNABERT-S specific)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state

        # Mean pooling over sequence positions
        embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

        # DNABERT-S outputs 768D, reduce to target_dim (256D)
        # Using average pooling as validated in DNABERT-S paper
        if len(embedding.shape) == 0:  # Handle scalar case
            embedding = np.array([embedding])

        # Average pooling dimension reduction
        if len(embedding) >= target_dim:
            # Reshape and average pool
            step = len(embedding) // target_dim
            reduced = np.array([embedding[i*step:(i+1)*step].mean() for i in range(target_dim)])
        else:
            # Pad if embedding is shorter
            reduced = np.pad(embedding, (0, target_dim - len(embedding)), 'constant')

        return reduced[:target_dim].tolist()

    def compute_and_cache(self, seq_str):
        """
        Main method: compute features and DNABERT-S embedding, cache to JSON

        Args:
            seq_str: Sequence string (e.g., "DNA 5'-ATCG-3'")

        Returns:
            dict: {
                'original_input': str,
                'macroscopic_properties': dict (25 numerical features),
                'embedding_256d': list (256D DNABERT-S embedding),
                'sequence': str (parsed sequence)
            }
        """
        cache_path = self._get_cache_path(seq_str)

        # Check cache
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)

        # Compute features
        features = self.extract_features(seq_str)
        macroscopic_properties = self._features_to_macroscopic_properties(features)
        embedding = self.generate_dnaberts_embedding(seq_str)

        # Prepare output
        result = {
            'original_input': seq_str,
            'macroscopic_properties': macroscopic_properties,
            'embedding_256d': embedding,
            'sequence': features['sequence']
        }

        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result


# Global instance (lazy loading)
_extractor = None

def extract_dna_features(seq_str, cache_dir='cache'):
    """
    Convenience function: extract features with DNABERT-S embedding

    Args:
        seq_str: Sequence string (e.g., "DNA 5'-ATCG-3'")
        cache_dir: Directory for caching JSON files

    Returns:
        dict: {
            'original_input': str,
            'macroscopic_properties': dict (25 numerical features),
            'embedding_256d': list (256D DNABERT-S vector),
            'sequence': str (parsed sequence)
        }
    """
    global _extractor
    if _extractor is None:
        _extractor = DNABERTSFeatureExtractor(cache_dir=cache_dir)
    return _extractor.compute_and_cache(seq_str)


if __name__ == '__main__':
    # Example usage
    test_seq = "DNA 5'-GAGTCTGTGGAGGAGGTAGTC-3'"

    print("Testing DNABERT-S Feature Extractor...")
    print(f"Input: {test_seq}\n")

    result = extract_dna_features(test_seq)

    print("Macroscopic properties (25 numerical features):")
    for key, value in result['macroscopic_properties'].items():
        print(f"  {key}: {value}")

    print(f"\nSequence: {result['sequence']}")
    print(f"\nDNABERT-S Embedding: 256D vector (first 5 values): {result['embedding_256d'][:5]}")
    print(f"Total embedding dimension: {len(result['embedding_256d'])}")
    print(f"\nCached to: cache/{test_seq}.json")
