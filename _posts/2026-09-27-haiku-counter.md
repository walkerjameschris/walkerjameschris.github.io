```py
#### Setup ####

#> uv init project && cd project
#> uv add nltk torch polars pyarrow plotnine umap-learn iprogress tqdm

import umap
import nltk
import torch
import polars as pl
import plotnine as p9
from nltk.corpus import cmudict

EPOCHS = 50
BATCH_SIZE = 256
MAX_SYLLABLES = 10
MAX_WORD_LENGTH = 15
EMBEDDING_DIMENSIONS = 64
LSTM_HIDDEN_DIMENSIONS = 128

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

letters = set("".join(words))
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

#### Visualize Clusters ####

# TODO: Revisit this
embeddings = model.embedding(X).to("cpu").detach().numpy()
cluster = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

#### Test with Haiku! ####

model.eval()

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
```
