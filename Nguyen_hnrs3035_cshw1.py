import re
import random
import pickle
import argparse
from collections import defaultdict
from collections import Counter

def default_dict():
    return defaultdict(int)

class NGramModel():
    def __init__(self, n: int):
        if n == 1 or n == 2:
            self.n = n
        self.vocabulary = set()
        self.probabilities = defaultdict(default_dict)
        self.occurrences = defaultdict(int)

    """
    Given a passage, train() calculates all probabilities of a token following previous token(s)
    """
    def train(self, passage: str):
        # Split full words or non-whitespace non-word characters
        split_words = re.findall(r"\w+|[^\w\s]", passage.lower())
        self.vocabulary = {(word,) for word in split_words}

        # Track previous token(s) of every token
        for i in range(self.n, len(split_words)):
            text_input = tuple(split_words[i - self.n:i])
            curr_word = split_words[i]
            self.probabilities[text_input][curr_word] += 1
            self.occurrences[text_input] += 1
            if self.n > 1:
                self.vocabulary.add(text_input)

        for text_input in self.probabilities:
            for possible_word in self.probabilities[text_input]:
                self.probabilities[text_input][possible_word] /= self.occurrences[text_input]

    """
    Predict next word based on is_deterministic() output
    """
    def predict_next_token(self, word, deterministic=False):
        if word not in self.vocabulary:
            return "<UNK>"

        if deterministic:
            return max(self.probabilities[word], key=self.probabilities[word].get)
        else:
            # If not deterministic, sample from full vocab using prob as weights
            choices = list(self.probabilities[word].keys())
            weights = list(self.probabilities[word].values())
            return random.choices(choices, weights=weights, k=1)[0]

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

"""
Splits individual characters of text into tokens, pairing most frequent tokens
"""
class BPE_algorithm():
    def __init__(self):
        self.vocabulary = set()
        self.vocab_to_index = {}

    """
    Develop vocabulary for a string, starting with individual chars and pairing the most frequent token(s) k times
    """
    def train(self, corpus: str, k=500):
        split_chars = list(corpus)
        self.vocabulary = set(corpus)

        for iteration in range(k):
            pair_count = defaultdict(int)

            for i in range(len(split_chars) - 1):
                pair_count[(split_chars[i], split_chars[i + 1])] += 1

            if not pair_count:
                break

            highest_freq = max(pair_count, key=pair_count.get)
            frequency = pair_count[highest_freq]
            new_token = ''.join(highest_freq)
            self.vocabulary.add(new_token)

            merged_tokens = []
            i = 0
            while i < len(split_chars):
                if i < len(split_chars) - 1 and (split_chars[i], split_chars[i + 1]) == highest_freq:
                    merged_tokens.append(new_token)
                    i += 2    # Skip next char because its part of token pair
                else:
                    merged_tokens.append(split_chars[i])
                    i += 1
            split_chars = merged_tokens

            print(f"Iteration {iteration + 1}: Highest pair {highest_freq} -> New token '{new_token}', Frequency {frequency}")

        self.token_to_id()
        return split_chars

    """
    Create a map for every token to its ID
    """
    def token_to_id(self):
        self.vocabulary.add("<UNK>")
        self.vocab_to_index = {word: idx + 1 for idx, word in enumerate(sorted(self.vocabulary))}    # Map indices of each vocab word
        self.vocab_to_index["<UNK>"] = 0
        return

    """
    Convert split string into its' IDs using greedy algorithm
    """
    def tokenize(self, text: str):
        tokens = []
        left = 0
        while left < len(text):
            match = None
            # Find longest token in given text
            for right in range(len(text), left, -1):
                sub = text[left:right]
                if sub in self.vocabulary:
                    match = sub
                    break
            if match is None:
                tokens.append("<UNK>")
                left += 1
            else:
                tokens.append(match)
                left += len(match)

        # Map created tokens into their IDs
        token_ids = [self.vocab_to_index.get(token, 0) for token in tokens]
        return tokens, token_ids

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

"""
Command line functions to train and use ngram and bpe models
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('activity', choices=['train_ngram', 'predict_ngram', 'train_bpe', 'tokenize'])
    parser.add_argument('--data', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--load', type=str)
    parser.add_argument('--word', type=str)
    parser.add_argument('--nwords', type=int)
    parser.add_argument('--text', type=str)
    parser.add_argument('--n', type=int, choices=[1, 2])
    parser.add_argument('--d', action='store_true')
    parser.add_argument('--k', type=int, default=500)

    args = parser.parse_args()

    if args.activity == 'train_ngram':
        if not args.data or not args.save or args.n is None:
            print("Please specify the data path and save path")
            return

        with open(args.data, 'r', encoding='utf-8') as f:
            data_arg = f.read()
        model = NGramModel(n=args.n)
        model.train(data_arg)
        model.save(args.save)
        print(f"{args.n}-gram model trained and saved at {args.save}.")

    elif args.activity == 'predict_ngram':
        if not args.load or not args.word or args.nwords is None:
            print("Please specify the number of words and load path")
            return

        model = NGramModel.load(args.load)
        if args.word not in model.vocabulary and args.n == 1:
            print("Word not in vocabulary")
            return

        if model.n == 1:
            current_word = tuple(args.word.split())
            output_text = list(current_word)

            for _ in range(args.nwords):
                next_token = model.predict_next_token(current_word, deterministic=args.d)
                output_text.append(next_token)
                current_word = tuple(next_token.split())

            print("Generated text:", " ".join(output_text))

        elif model.n == 2:
            current_word = tuple(args.word.split())
            output_text = list(current_word)

            for _ in range(args.nwords):
                next_token = model.predict_next_token(current_word, deterministic=args.d)
                output_text.append(next_token)
                current_word = tuple(output_text[-2:])

            print("Generated text:", " ".join(output_text))

    elif args.activity == 'train_bpe':
        if not args.data or not args.save:
            print("Please specify the data path and save path")
            return

        with open(args.data, 'r', encoding='utf-8') as f:
            data_arg = f.read()
        bpe_model = BPE_algorithm()
        bpe_model.train(data_arg, k=args.k)
        bpe_model.save(args.save)
        print(f"BPE model trained and saved at {args.save}.")

    elif args.activity == 'tokenize':
        if not args.load or not args.text:
            print("Please specify the load path and text to tokenize")
            return

        bpe_model = BPE_algorithm.load(args.load)
        tokens, token_ids = bpe_model.tokenize(args.text)
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)

    else:
        raise ValueError('Unknown activity')

if __name__ == "__main__":
    main()