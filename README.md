This is a n-gram model and a BPE tokenizer exploring the fundamentals of LLMs. 

The models can be trained and tested on the MobyDick.txt included in the repo, using the built-in command line interface.

The first positional (required) argument should be a selector for which activity to
perform. Options should be “train_ngram”, “predict_ngram”, “train_bpe”, and
“tokenize”

An argument (--data) that points to the path (including filename and extension) to
the training data corpus. This will only be used if the user selects the
“train_ngram” or “train_bpe” activity.

An argument (--save) that points to the path where the ngram or BPE model
(depending on which activity was chosen) will be saved so that it can be loaded in
the “predict_ngram” or “tokenize” activity. The model can be serialized and saved
using pickle.

An argument (--load) that points to the path where the trained ngram or BPE
model was saved (depending on which activity was chosen). The model object can be
loaded using pickle.

A string argument (--word) that specifies the first word (or words) used for the
“predict_ngram” activity.

An integer argument (--nwords) that specifies the number of words to predict for
the “predict_ngram” activity.

A string argument (--text) that specifies the string to be tokenized through the
“tokenize” activity.

An integer arugment (--n) that specifies the order of the ngram (choices should be 1
or 2). This should only be needed for the “train_ngram” activity.

An arugment (--d) that set the deterministic flag for the predict_next_word() method
of of the ngram model to True. This should only be used for the “predict_ngram”
activity.
