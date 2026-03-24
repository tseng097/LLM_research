import tiktoken
import sys
import os
import math
import random
from collections import Counter
class NGramModel:
    def __init__(self, n, vocab_size, use_interpolation=True, k=0.5):
        self.n = n
        self.counts = {i: Counter() for i in range(1, n + 1)}
        self.vocab_size = vocab_size
        self.use_interpolation = use_interpolation  
        self.k = k 
    def train(self, tokens):
        for i in range(1, self.n + 1):
            for j in range(len(tokens) - i + 1):
                ngram = tuple(tokens[j:j+i])
                self.counts[i][ngram] += 1

    def calculate_perplexity(self, tokens):
        if not tokens: return float('inf')
        
        log_prob_sum = 0
        n_count = 0
        k = self.k  
        total_tokens = sum(self.counts[1].values())

        for i in range(len(tokens)):
            if self.use_interpolation: #  Interpolation
                weights = [0.6,0.3,0.1] if self.n == 3 else [1.0/self.n]*self.n
                prob = 0
                for order in range(1, self.n + 1):
                    if order == 1:
                        c_ngram = self.counts[1].get((tokens[i],), 0)
                        p = (c_ngram + k) / (total_tokens + k * self.vocab_size)
                    elif i >= order - 1:
                        ctx = tuple(tokens[i - order + 1 : i])
                        c_ngram = self.counts[order].get(ctx + (tokens[i],), 0)
                        c_ctx = self.counts[order - 1].get(ctx, 0)
                        p = (c_ngram + k) / (c_ctx + k * self.vocab_size)
                    else: p = 0
                    prob += weights[order - 1] * p
            else: # No Interpolation
                order = min(i + 1, self.n) 
                if order == 1:
                    c_ngram = self.counts[1].get((tokens[i],), 0)
                    prob = (c_ngram + k) / (total_tokens + k * self.vocab_size)
                else:
                    ctx = tuple(tokens[i - order + 1 : i])
                    c_ngram = self.counts[order].get(ctx + (tokens[i],), 0)
                    c_ctx = self.counts[order - 1].get(ctx, 0)
                    prob = (c_ngram + k) / (c_ctx + k * self.vocab_size)
            
            if prob > 0:
                log_prob_sum += math.log(prob)
                n_count += 1
        
        return math.exp(-(log_prob_sum / n_count)) if n_count > 0 else float('inf')

class AuthorClassifier:
    def __init__(self, n=3, k=0.5):
        self.n = n
        self.k = k  
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.models = {}
        self.global_vocab = set()
        self.local_stats = {}

    def build_global_vocab(self, file_paths): #normalize
        print("Building global vocabulary and analyzing local stats...")
        for path in file_paths:
            filename = os.path.basename(path)
            with open(path, 'r', encoding='utf-8') as f:
                tokens = self.tokenizer.encode(f.read())
                unique_tokens = set(tokens)
                self.local_stats[filename] = {
                    "total": len(tokens),
                    "unique": len(unique_tokens)
                }
                self.global_vocab.update(unique_tokens)
        

        # for fname, stats in self.local_stats.items():
        #     print(f"[{fname}] Total: {stats['total']}, Unique: {stats['unique']}")
        # print(f"Combined Global Vocab Size (|V|): {len(self.global_vocab)}\n")

    def classify(self, tokens):
        results = {author: model.calculate_perplexity(tokens) 
                   for author, model in self.models.items()}
        return min(results, key=results.get)

def get_data_randomly(file_path, ratio=0.1):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    random.seed(42)
    random.shuffle(lines)
    split_point = int(len(lines) * (1 - ratio))
    return lines[:split_point], lines[split_point:]
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 classifier.py authorlist [-test testfile]")
        sys.exit(1)
   
    authorlist_path = sys.argv[1]
    test_mode = "-test" in sys.argv
    data_dir = os.path.join("Corpora_export", "Authorship")
    
    classifier = AuthorClassifier(n=3, k=0.5) 
    author_files = [line.strip() for line in open(authorlist_path, 'r', encoding='utf-8') if line.strip()]
    full_paths = [os.path.join(data_dir, f) for f in author_files]

    classifier.build_global_vocab(full_paths)
    v_size = len(classifier.global_vocab)

    print("Training models...")
    dev_data_by_author = {}
    for file_name in author_files:
        author_name = file_name.split('_')[0]
        full_path = os.path.join(data_dir, file_name)

        model = NGramModel(n=classifier.n, vocab_size=v_size, k=classifier.k)
        
        if not test_mode:
            train_lines, dev_lines = get_data_randomly(full_path, ratio=0.1)
            train_tokens = classifier.tokenizer.encode(" ".join(train_lines))
            dev_data_by_author[author_name] = [classifier.tokenizer.encode(l) for l in dev_lines]
        else:
            with open(full_path, 'r', encoding='utf-8') as f:
                train_tokens = classifier.tokenizer.encode(f.read())
        
        model.train(train_tokens)
        classifier.models[author_name] = model

    if not test_mode:

        for mode in [False, True]:
            mode_name = "Model B (Interpolation ON)" if mode else "Model A (Interpolation OFF)"
            print(f"\n--- Running Evaluation: {mode_name} ---")
            
            for model in classifier.models.values():
                model.use_interpolation = mode
            
            total_correct = 0
            total_sentences = 0
            
            for author, sentences in dev_data_by_author.items():
                correct = sum(1 for s in sentences if classifier.classify(s) == author)
                total_correct += correct
                total_sentences += len(sentences)
                acc = (correct / len(sentences)) * 100 if sentences else 0
                print(f"{author:15} Accuracy: {acc:.1f}%")
            
            print(f">> Overall {mode_name} Accuracy: {(total_correct/total_sentences)*100:.2f}%")
            
    else:
        for mode in [False, True]:  
            for model in classifier.models.values():
                model.use_interpolation = mode
                
            test_file_path = sys.argv[sys.argv.index("-test") + 1]
            from collections import Counter
            prediction_stats = Counter()
            total_lines = 0

            print(f"\n Analyzing test file: {test_file_path} with {'Interpolation ON' if mode else 'Interpolation OFF'}")
            with open(test_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        tks = classifier.tokenizer.encode(line.strip())
                        winner = classifier.classify(tks) 
                        print(winner)
                        prediction_stats[winner] += 1
                        total_lines += 1
                       
            
            print("\n" + "="*30)
            print(f"Classification Distribution (Total: {total_lines} lines)")
            print("-"*30)
            for author, count in prediction_stats.most_common():
                percentage = (count / total_lines) * 100
                print(f"{author:15}: {count:4} times ({percentage:.1f}%)")
            print("="*30)
    
