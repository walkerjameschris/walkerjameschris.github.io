---
title: "Using Deep Learning to Count Syllables"
date: "2026-09-27"
---

I enjoy writing haiku poems and I also enjoy deep learning. So
I thought, what about building a deep learning model to predict
the number of syllables in a word! While it is reasonable to use
a lookup table to determine the syllables in a word, building a
model allows us to check syllable counts for haiku poems to
leverage proper nouns and new words not available in a lookup.

## TL;DR: Try the Model!

Here is the model running client side in your web browser using
pure HTML/JS and the ONNX standard! This is a 500k+ parameter
deep learning model with an `nn.Embedding` layer, an `nn.LSTM`,
and text parsing in *your browser*. It even works with made-up
words (e.g., splorfinating or flumfloxed):

<iframe src="../assets/syllable-counter.html" width="100%" height="200" frameborder="0" scrolling="no"></iframe>

Please note:
- Currently, this only supports one word at a time in browser, I may
  expand in the future to multiple words (or a complete haiku) in JS.
- Absolutely **no** data is retained in the textbox above. The entire
  model and website are client side. View the source code
  [here](https://github.com/walkerjameschris/walkerjameschris.github.io).


## Data

To build my model, I am using `nltk` and the `cmudict` dataset.
This dataset of over 100k words includes pronunciations for
each word. We can parse these pronunciations to build targets
for our model.

For example, `apple` is represented by the pronunciation below.
We count the number of times a fragment ends with a digit, so
`AE1` and `AH0` end with integers. So there are two syllables
in this word!

```py
['AE1', 'P', 'AH0', 'L']
```

---

We repeat this for every word and then we obtain the "truth"
values for our model. Now how do we feed words into a model?
We are going to build a deep learning model which means we need
to create *numeric `tensor` representations* of words.

To keep things reasonable, I limited the word *length* to 15
characters. So let's take `apple` again. We will represent
`apple` as an encoding. Since "a" is the first letter, it gets a
`1`, and "p" is the sixteenth letter, so it gets a `16`, and
so on. Then, for any unused characters we give them a `0`.

```py
[1, 16, 16, 12, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

We repeat this for all words and then we have the inputs (word
encodings) and outputs (syllable counts). I divided the data into
train and test using `train_test_split` from `sklearn`.

## Model Architecture and Training

Our model relies on two main components:
1. An Embedding Layer: Computes dense vector representations for
   raw character indices. Think of this like giving the model a rich
   map of letter identities rather than forcing it to reason over plain
   integers.
2. A Bidirectional LSTM (Long Short-Term Memory): Learns sequential
   patterns across the letters of a word by reading them both forward and
   backward simultaneously. This dual-direction context is crucial for
   capturing English spelling quirks (like trailing silent "e"s or
   vowel clusters) before passing the final state to a linear layer
   to predict syllable count!

I decided to use an LSTM for the core of this model (as opposed to a
a simple feedforward neural network) given the ubiquity for LSTMs in
[small NLP projects](https://ieeexplore.ieee.org/document/11485794).
Moreover, because I am going to serve this model client-side the final
binary needs to be compact so a large transformer architecture is
likey overkill for this use case. I would try more architectures in
an enterprise model development setting.

We define this model using `torch.nn.Module` and define `embedding`
and `lstm` members. We also define the `forward` pass as is standard
practice:

```py
class SyllableClassifier(torch.nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_DIMENSIONS,
            padding_idx=0
        )
        
        self.lstm = torch.nn.LSTM(
            input_size=EMBEDDING_DIMENSIONS,
            hidden_size=LSTM_HIDDEN_DIMENSIONS,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = torch.nn.Linear(
            in_features=LSTM_HIDDEN_DIMENSIONS * 2,
            out_features=MAX_SYLLABLES + 1
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :]) # Last character timestep
```

---

Using this model, we define a standard training loop including a
loss function (Cross Entropy), and an ADAM optimizer. This is
then looped over a defined number of `EPOCHS`. Note that we 
divide the training data into *batches* using `torch.split`:

```py
for epoch in range(EPOCHS):
    
    batches = zip(
        torch.split(X, BATCH_SIZE),
        torch.split(y, BATCH_SIZE)
    )
    
    for X_batch, y_batch in batches:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.4f}")
```

## Evaluation

First, I wanted to see how often the model was correct on train
and test. Turns out, its pretty good! The model correctly guessed
the number of syllables over 95% of the time for both train and test:

| Population | % Correct |
| - | - |
| Train | 99.78% |
| Test | 95.54% |

From here, we can try out our model! I defined a helper function
which allows a user to pass a complete haiku and obtain results.
Consider these three haiku poems:

**Coffee:**
> Coffee aroma \
> Flows freely from room to room \
> Sunlight pours inside

**Mist:**
> Dark misty hillsides \
> Damp ferns hidden in the fog \
> Water drips on moss

**Weird:**
> This is six syllables \
> Which is the wrong amount \
> For a five seven five haiku

Which turns into the result below. Recall that a haiku follows
a strict 5, 7, 5 syllable pattern:

```py
coffee: [5, 7, 5]
mist: [5, 7, 5]
weird: [6, 6, 8]
```

It even works for *new* made up words like *Cridget* returns 2
as it is pronounced *Cri-dget*!

## Limitations

- I would want to set up more advanced random seed setting for
  both data and model training for reproducibility
- I could do more advanced sampling and performance analysis
  (e.g., where does the model slip up and for what type of words?)
- This is all one script; an enterprise model setting would likely
  require a multi file construct
- I could add type hints and unit tests for all functions; additionally
  the helper functions are not *pure* meaning they require global
  state which I would also likely change
- I select the *last* timestep from the LSTM whereas I might be
  able to grab the last non-null character

## Code

Here is the complete end to end code. Note that it was run within
a `uv` environment with a GPU. However, even on CPU, this model
is small enough to converge in a few minutes (maybe 5-15):

```py
#### Setup ####

import nltk
import torch
from nltk.corpus import cmudict
from sklearn.model_selection import train_test_split

EPOCHS = 50
BATCH_SIZE = 256
MAX_SYLLABLES = 10
MAX_WORD_LENGTH = 15
EMBEDDING_DIMENSIONS = 64
LSTM_HIDDEN_DIMENSIONS = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

#### Extract Words ####

nltk.download("cmudict")
dictionary = cmudict.dict()

words = []
syllables = []

for word, data in dictionary.items():

    # Words must be within the max length
    if len(word) < 1 or len(word) > MAX_WORD_LENGTH:
        continue

    # Words must ONLY be letters, no contractions
    if not word.isalpha():
        continue

    # Here data is a list of technical representations
    # of possible pronunciations for a word. We only
    # will take the first pronunciation. Each word
    # pronunciation contains components. If one such
    # component ends with a number, that indicates
    # a syllable. For example, apple is represented
    # by ['AE1', 'P', 'AH0', 'L']. Elements 0 and 2
    # end with a number indicating 2x syllables!
    n = len([i for i in data[0] if i[-1].isdigit()])

    # Words must have a reasonable amount of syllables
    if n >= 1 and n <= MAX_SYLLABLES:
        words.append(word)
        syllables.append(n)

#### Data Quality Check ####

assert len(words) == len(syllables)
assert max(syllables) <= MAX_SYLLABLES
assert max(len(i) for i in words) <= MAX_WORD_LENGTH

#### Build Dataset ####

letters = "abcdefghijklmnopqrstuvwxyz"
char_to_idx = {v: i + 1 for i, v in enumerate(letters)}
char_to_idx[""] = 0 # This is the NULL padding index

#### Create Training Tensors #####

# This function would error for unsanitized string inputs
# containing punctuation or capital letters. However, for
# this toy example, it will be deployed client-side in
# JavaScript within a textbox with input validation.
def encode(words):
    if not isinstance(words, list):
        words = [words]
    encodings = []
    for word in words:
        encoding = MAX_WORD_LENGTH * [char_to_idx[""]]
        for i, letter in enumerate(word):
            encoding[i] = char_to_idx[letter]
        encodings.append(encoding)
    return encodings

X = torch.tensor(encode(words), dtype=torch.long).to(device)
y = torch.tensor(syllables, dtype=torch.long).to(device)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

#### Model Architecture ####

class SyllableClassifier(torch.nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_DIMENSIONS,
            padding_idx=0
        )
        
        self.lstm = torch.nn.LSTM(
            input_size=EMBEDDING_DIMENSIONS,
            hidden_size=LSTM_HIDDEN_DIMENSIONS,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = torch.nn.Linear(
            in_features=LSTM_HIDDEN_DIMENSIONS * 2,
            out_features=MAX_SYLLABLES + 1
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

#### Train Model ####

model = SyllableClassifier(len(char_to_idx)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    
    batches = zip(
        torch.split(X_train, BATCH_SIZE),
        torch.split(y_train, BATCH_SIZE)
    )
    
    for X_batch, y_batch in batches:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.4f}")

model.eval()

#### Measure Train/Test Performance ####

@torch.no_grad()
def evaluate(X, y):
    batches = torch.split(X, BATCH_SIZE)
    logits = [model(i) for i in batches]
    predictions = torch.vstack(logits).argmax(dim=1)
    return torch.mean(100 * (predictions == y).float()).item()
           
print(f"Train: {evaluate(X_train, y_train):.2f}% correct")
print(f"Test: {evaluate(X_test, y_test):.2f}% correct")

#### Test with Haikus! ####

@torch.no_grad()
def count_haiku(poem):
    result = []
    for line in poem:
        n_line = 0
        for word in line.split(" "):
            encodings = encode(word.lower())
            word_input = torch.LongTensor(encodings).to(device)
            n_line += model(word_input).argmax(dim=1).item()
        result.append(n_line)
    return result

poems = {
    "coffee": [
        "Coffee aroma",
        "Flows freely from room to room",
        "Sunlight pours inside"
    ],
    "mist": [
        "Dark misty hillsides",
        "Damp ferns hidden in the fog",
        "Water drips on moss"
    ],
    "weird": [
        "This is six syllables",
        "Which is the wrong amount",
        "For a five seven five haiku"
    ]
}

for poem, contents in poems.items():
    print(f"{poem}: {count_haiku(contents)}")

#### Export Model ####

ex_input = torch.zeros((1, MAX_WORD_LENGTH), dtype=torch.long, device=device)

torch.onnx.export(
    model,
    ex_input,
    "syllable-classifier.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input_words"],
    output_names=["syllable_logits"],
    dynamo=False
)
```
