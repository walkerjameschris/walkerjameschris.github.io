---
title: "Using Deep Learning to Count Syllables for Haikus"
date: "2026-09-27"
---

I enjoy writing haiku poems and I also enjoy deep learning. So
I thought, what about building a deep learning model to predict
the number of syllables in a word! While it is reasonable to use
a lookup table to determine the syllables in a word, building a
model allows us to check syllable counts for haiku poems to
leverage proper nouns and new words not available in a lookup.

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
encodings) and outputs (syllable counts).

## Architecture

Our model is two major components:
1. An embedding component (to convert encodings to latent vectors)
2. Long-Short-Term-Memory or LSTM (to learn the relationships
   *across* the letters of a word, both backwards and forwards.

```py
#### Setup ####

#> uv init project && cd project
#> uv add nltk torch polars pyarrow plotnine umap-learn numpy

import umap
import nltk
import torch
import numpy as np
import polars as pl
import plotnine as p9
from nltk.corpus import cmudict

EPOCHS = 50
BATCH_SIZE = 256
MAX_SYLLABLES = 10
MAX_WORD_LENGTH = 15
EMBEDDING_DIMENSIONS = 64
LSTM_HIDDEN_DIMENSIONS = 128

# These are for visualizing specific words on the
# clustering plot even if they aren't included
# based on the "every-N" downsample
EVERY_N = 50
EXAMPLE_CLUSTERING_WORDS = ["apple", "grapple", "grape"]
EXAMPLE_CLUSTERING_SYLLABLES = [2, 2, 1]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

#### Extract Words ####

nltk.download("cmudict", quiet=True)
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

    # Words must have a resonable amount of syllables
    if n >= 1 and n <= MAX_SYLLABLES:
        words.append(word)
        syllables.append(n)

#### Data Quality Check ####

assert len(words) == len(syllables)
assert max(syllables) <= MAX_SYLLABLES
assert max(len(i) for i in words) <= MAX_WORD_LENGTH

#### Build Dataset ####

letters = sorted(set("".join(words)))
char_to_idx = {v: i + 1 for i, v in enumerate(letters)}
char_to_idx[""] = 0 # This is the NULL padding index

#### Helper Functions #####

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

X = torch.LongTensor(encode(words)).to(device)
y = torch.LongTensor(syllables).to(device)

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
        return self.fc(lstm_out[:, -1, :]) # Last character timestep

#### Train Model ####

model = SyllableClassifier(len(char_to_idx)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

model.eval()

#### Visualize Clusters ####

# Pass cluster sample through model
with torch.no_grad():
    cluster_words = words[0::EVERY_N] + EXAMPLE_CLUSTERING_WORDS
    cluster_syllables = syllables[0::EVERY_N] + EXAMPLE_CLUSTERING_SYLLABLES
    X_cluster = torch.LongTensor(encode(cluster_words)).to(device)
    cluster_output = model(X_cluster).detach().cpu().numpy()

# Establish UMAP clustering model
cluster_umap = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    n_jobs=1
)

# Create data frame for plotting
cluster_sample = (
    pl.DataFrame(
        cluster_umap.fit_transform(cluster_output),
        schema=["Dimension 1", "Dimension 2"]
    )
    .with_columns(
        words=pl.Series(cluster_words),
        syllables=pl.Series(cluster_syllables)
    )
    .unique()
)

plot = (
    p9.ggplot(
        data=cluster_sample,
        mapping=p9.aes(
            x="Dimension 1",
            y="Dimension 2",
            color = "factor(syllables)"
        )
    ) +
    p9.geom_point() +
    p9.geom_label(
        mapping=p9.aes(label="words"),
        data=cluster_sample.filter(
            pl.col("words").is_in(EXAMPLE_CLUSTERING_WORDS)
        )
    )
)

plot.show()

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
    ],
    "rey": [
        "Rey is a special boy",
        "Cridget"
    ]
}

for poem, contents in poems.items():
    print(f"{poem}: {count_haiku(contents)}")

```
