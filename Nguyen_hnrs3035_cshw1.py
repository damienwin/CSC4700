import re
import random
from collections import defaultdict
from collections import Counter

class NGramModel():
    def __init__(self, n:int):
        if n == 1 or n == 2:
            self.n = n
        self.vocabulary = set()
        self.probabilities = defaultdict(lambda: defaultdict(int))
        self.occurrences = defaultdict(int)

    """
    Given a passage, train() calculates all probabilities of a token following previous token(s)
    """
    def train(self, passage: str):
        # Split full words or non-whitespace non-word characters
        split_words = re.findall(r"\w+|[^\w\s]", passage.lower())
        print(split_words)
        self.vocabulary = set(split_words)

        # Track previous token(s) of every token
        for i in range(self.n, len(split_words)):
            text_input = tuple(split_words[i-self.n:i])
            curr_word = split_words[i]
            self.probabilities[text_input][curr_word] += 1
            self.occurrences[text_input] += 1
            if self.n > 1:
                self.vocabulary.add(text_input)

        for text_input in self.probabilities:
            for possible_word in self.probabilities[text_input]:
                self.probabilities[text_input][possible_word] /= self.occurrences[text_input]

    """
    Determine the most probabilistic token(s)
    """
    def is_deterministic(self, word):
        if word in self.vocabulary:
            word_choices = []
            maxprob = max(self.probabilities[(word,)].values())
            print(f"MAXPROB: {maxprob}")
            for possible_word in self.probabilities[(word,)]:
                if self.probabilities[(word,)][possible_word] == maxprob:
                    word_choices.append(possible_word)

            deterministic = False
            if len(word_choices) == 1:
                deterministic = True

            return deterministic, word_choices

        else:
            print("Error: word not in vocabulary")
            return

    """
    Predict next word based on is_deterministic() output
    """
    def predict_next_token(self, word, deterministic=False):
        deterministic, word_choices = self.is_deterministic(word)
        if deterministic:
            return word_choices[0]
        else:
            return random.choices(word_choices)

class BPE_algorithm():
    def __init__(self, model:NGramModel):
        self.vocabulary = set()
        self.vocab_to_index = defaultdict()

    def train(self, corpus:str, k=500):
        split_chars = list(corpus)
        self.vocabulary = set(corpus)

        for iteration in range(k):
            pair_count = defaultdict(int)

            for i in range(len(split_chars)-1):
                pair_count[(split_chars[i], split_chars[i+1])] += 1

            if not pair_count:
                break

            highest_freq = max(pair_count, key=pair_count.get)
            new_token = ''.join(highest_freq)
            self.vocabulary.add(new_token)

            merged_tokens = []
            i = 0
            while i < len(split_chars):
                if i < len(split_chars) - 1 and (split_chars[i], split_chars[i + 1]) == highest_freq:
                    merged_tokens.append(new_token)
                    i += 1
                else:
                    merged_tokens.append(split_chars[i])
                i += 1
            split_chars = merged_tokens
        return split_chars


    def token_to_id(self):
        self.vocabulary.add("<UNK>")
        self.vocab_to_index = {word: idx + 1 for idx, word in enumerate(sorted(self.vocabulary))}
        self.vocab_to_index["<UNK>"] = 0
        return

    def tokenize(self, split_chars):
        tokenized_corpus = [self.vocab_to_index.get(token, 0) for token in split_chars]
        return tokenized_corpus

passage = "Hello, world! This is a test. Hello again!"
print(passage)
print(set(passage))
Model = NGramModel(1)
Model.train(passage)
print(Model.vocabulary)
print(dict(Model.probabilities))
word = "hello"
print(Model.probabilities[(word,)].items())
print(Model.is_deterministic(word))
print(Model.predict_next_token(word))

bpe_model = BPE_algorithm(Model)
tokenized_words = bpe_model.train(passage)
print(tokenized_words)
print(bpe_model.vocabulary)
bpe_model.token_to_id()
print(bpe_model.tokenize(tokenized_words))
print(bpe_model.vocab_to_index)






