---
title: "Using an LLM and Embeddings to Study Collective Cultures in History"
date: "2026-09-09"
---

## Collectivism and Individualism

The United States is a decidedly *individualistic* culture. This is a cultural
trait common in Western nations like Canada, the UK, and Australia. In an
individualistic culture, the fundamental unit is unsurprisingly, the individual.
While individuals in these cultures do care about their families and friends,
the goals of an individual are prioritized over the goals of the group:
- Whom will *I* marry?
- Where will *I* study?
- Where will *I* live?
- What will *I* do for work?

As someone who spends a lot of time reading Scripture, I am increasingly aware
of the *collective* nature of ancient Biblical societies. This means that while
individual actions are important, the family and culture in which a person lives
is where they find their identity. Thus, the goals of the group are prioritized
over the goals of the individual:
- How will my marriage provide benefit (by way of joining families; alliances) our *tribe*?
  How will this *honor our* relatives?
- What knowledge and wisdom can I provide to my *family*?
- Where will *we* live?
- How will my occupation strengthen our *group*? Does this promote group harmony?

For example, the Bible exists in ancient agrarian, collective, and high-context
societies where family lineage, honor/shame, and *reading between the lines* was
how the cultures of the Bible operate. Conversely, America in the industrial
revolution was industrial, individualist, and low-context; business tycoons and
individuals raced to build their own business empires often in *spite* of their
upbringing (e.g., rags to riches). However, the Deep South (particularly in early
America) is somewhere in between where American individualism was taking root,
but it still largely existed in an agrarian (cotton, tobacco) and high-context
setting; anyone who has been told "bless your heart" knows that it can mean
*many* different things based on the tones and context...

## Related Traits

I hypothesized that as societies move from nomadic/agrarian to industrial
that they would shed some of their collectivism in favor of individualism. Moreover
in an individualistic society, people are generally more explicit and direct in
speech (this is called low-context where the speaker assumes the listener has
little knowledge of how the speaker thinks) as opposed to high-context where there
is lots of reading between the lines. This brings about anthropological "poles"
I am interested in studying. My research question is as follows:

> To what extent do agrarian economic structures predict the coexistence of
> collectivist cultural values and high-context communication styles?

## Methods

To determine these relationships, I want to measure the correlation between three
anthropological poles:
- Collectivism *to* Individualism
- Agrarianism *to* Industrialization
- High-Context *to* Low-Context Communication

I selected 5 texts available on Project Gutenberg as `.txt` files which I can chunk
and embed using an embedding model:

| Book | Summary | Expected Pole Alignment |
| ---- | ------- | ----------------------- |
| [KJV Bible](https://www.gutenberg.org/cache/epub/10/pg10.txt) | Scripture, lots of subtext and cultural influence, honor-shame | Agrarian, Collective, and High-Context |
| [The Southerner](https://www.gutenberg.org/cache/epub/15865/pg15865.txt) | Early 20th-century Southern text tracking family legacy and honor/shame | Agrarian, Collective, and High-Context |
| [O Pioneers](https://www.gutenberg.org/cache/epub/2426/pg2426.txt) | Midwestern frontier farming completely stripped of collective safety nets, highlighting radical self-reliance. | Agrarian, Individualist, and Low-Context |
| [The Jungle](https://www.gutenberg.org/cache/epub/140/pg140.txt) | Chicago slaughterhouses; focuses heavily on the mechanics of industrial wages and the explicit push for labor unions. | Industrial, Collective, and Low-Context |
| [The Iron Heel](https://www.gutenberg.org/cache/epub/1164/pg1164.txt) | Set in a gritty industrial future. London focuses aggressively on individual steel/rail oligarchs and raw corporate power. | Industrial, Individualist, and Low-Context |

These 5 texts are then chunked into segments of 500 words and passed to an *embedding model*,
specifically, `embeddinggemma` from Google. For the uninitiated, an embedding model is an essential
component of modern AI systems. Effectively, they are mathematical functions which accept
text of arbitrary length (e.g., a few paragraphs) and return a *vector* (a list of numbers)
of a fixed length. The more similar the list of numbers, the more similar the text (even
if the text is of different lengths).

- "The quick brown fox..." might become `[0.12, 0.34, 0.91]`
- "The brown quick fox..." might become `[0.11, 0.44, 0.89]`
- "A tall building was..." might become `[0.95, 0.01, 0.02]`

In an AI system, embeddings are used to determine which documents or web pages (out of 
thousands of potential documents) are relevant to the question. This both improves the
speed and reliability of AI systems. *Note:* this mechanism is known as retrieval 
augmented generation (RAG). Other systems exist, however, RAG remains very popular.

--- 

Using these embeddings from the texts above, I will compare them to statements discussing
*individualism*, *industrialization*, and *language context*. These statements are generated by
an LLM (specifically `gemma4` from Google) using the following prompt template where *description*
is a statement like "Traditional agrarian life, seasonal crop reliance, rural harvesting, land
ties, ancestral farming":

```
You are an expert cross-cultural anthropologist and linguist.

Generate exactly 10 distinct, highly descriptive phrases or
micro-scenarios that represent an extreme baseline for this trait:
{description}

CRITICAL RULES:
1. Isolate the concept. Make it pure, focused, and extreme.
2. Keep them brief (8-15 words each).
3. Return ONLY valid JSON matching the template below. 
4. Do not include any reasoning, conversational text,
   markdown formatting blocks, or chatter. Only raw JSON.
```

This prompt would generate responses like:

```json
{
  "phrases": [
    "Ancestral seeds dictate the planting cycle and harvest timing.",
    "The communal field dictates seasonal labor and land boundaries.",
    "Harvest rituals bind the family to the ancestral soil.",
    "Seasonal shifts determine the rhythm of the entire community.",
    "Land ties define kinship and the obligation to the earth.",
    "Crop rotation governs the annual cycle of rural existence.",
    "The ancestral path dictates the boundaries of the cultivated land.",
    "Harvesting is a sacred, seasonal obligation of the lineage.",
    "Farming is a cyclical, inherited practice of rural life.",
    "The land dictates the rhythm of the ancestral farming."
  ]
}
```

I then *embed* the statements above and compare them to the text fragments,
such as a passage from the KJV:

- **LLM Statement:** "Ancestral seeds dictate the planting cycle and harvest timing."
- **Verse from KJV:** "He that observeth the wind shall not sow; and he that regardeth the clouds shall not reap."

Using `embeddinggemma` these two texts have a similarity of `0.4` where scores range from -1 to 1.
Texts with a score of -1 are perfectly opposite where texts with a score of 1 are exactly the same.
A score of `0.4` means these texts are related and moderately similar (using cosine similarity):

```py
import ollama
import numpy as np

llm_statement = "Ancestral seeds dictate the planting cycle and harvest timing."
kjv_verse = "He that observeth the wind shall not sow; and he that regardeth the clouds shall not reap."

llm_embed = ollama.embed("embeddinggemma", llm_statement)
kjv_embed = ollama.embed("embeddinggemma", kjv_verse)

# Note that the dot-product of two normalized embeddings is cosine similarity
np.dot(
    np.array(llm_embed.embeddings),
    np.array(kjv_embed.embeddings).T
).item()
# 0.40
```

I then repeat this process across 10 statements across each of the three
"poles" for a total of 30 statements.

## Determining Scores for Each Text Fragment

| Spectrum | Formula |
| --- | --- |
| **Agrarian** | Avg(10 Agrarian) |
| **Collective** | Avg(10 Collective) |
| **Context** | Avg(10 High-Context) |

In the end, I obtain a table like this (this is a `polars` data frame). Note also that each `id`
or row corresponds to a 500 word text fragment in one of the 5 texts (e.g., the KJV or a passage
in The Jungle).

```
shape: (2_070, 4)
┌──────┬───────────┬────────────┬───────────┐
│ id   ┆ Agrarian  ┆ Collective ┆ Context   │
│ ---  ┆ ---       ┆ ---        ┆ ---       │
│ u32  ┆ f32       ┆ f32        ┆ f32       │
╞══════╪═══════════╪════════════╪═══════════╡
│ 1736 ┆ 0.012881  ┆ -0.015657  ┆ -0.028202 │
│ 280  ┆ 0.091485  ┆ 0.094038   ┆ 0.07678   │
│ …    ┆ …         ┆ …          ┆ …         │
│ 1179 ┆ 0.048913  ┆ 0.06191    ┆ 0.03391   │
│ 1816 ┆ -0.011104 ┆ -0.009879  ┆ -0.016108 │
└──────┴───────────┴────────────┴───────────┘
```

## Results

Using the similarity scores across over 2000 text fragments, I can determine the
Pearson correlation between each of the three combinations of scores. It turns out
that all three dimensions are correlated!

| Dimension A | Dimension B | Pearson Correlation |
| --- | --- | --- |
| Collective | Context | 0.32 |
| Agrarian | Context | 0.09 |
| Agrarian | Collective | 0.69 |

> The data supports the core hypothesis: agrarian economic structures consistently
> align with both collectivist values and high-context communication styles. However,
> the relationship between agrarian economic structures and high-context communication
> is *weak* at best (for example, the direct communication style of the agrarian Midwest)
> This is further supported by the visualization below:

![](/assets/culture-corr.png)

## Limitations

- It's worth noting that individualistic cultures exhibit collectivism (e.g. sports teams,
  state pride, nationalism) and collective cultures exhibit individualism (selfishness,
  individual pride) but thinking of cultures along a spectrum of collectivism and individualism
  is helpful as a mechanism.
- Embedding models and language models are used to generate the results; while useful they
  may contain inherent biases and limitations for generating and measuring text similarity.
- The selection of 5 texts is intentional, but limited. A more robust study would include
  more texts.
- I could have included a "control pole" which should be *totally* unrelated to the three
  anthropological dimensions that should have a near-zero correlation.
- Given the relatively small sample size (~2000 observations) I could have bootstrapped the
  results or used more chunks to generate a confidence interval.

## Code

*Note:* this was built using a `uv` virtual environment. It also assumes an Ollama server
is running with the `OLLAMA_HOST` environment variable is set with `embeddinggemma` and
`gemma4:e2b` is loaded. A GPU is *strongly* encouraged for the Ollama instance.

```py
#### Setup ####

import re
import json
import tqdm
import ollama
import requests

import numpy as np
import polars as pl
import plotnine as p9

from functools import reduce
from numpy.typing import NDArray

CHUNK_WORDS = 500

EMBEDDING_MODEL = "embeddinggemma"
LANGUAGE_MODEL = "gemma4:e2b"

URLS = {
    "Bible KJV": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "The Southerner": "https://www.gutenberg.org/cache/epub/15865/pg15865.txt",
    "O Pioneers": "https://www.gutenberg.org/cache/epub/2426/pg2426.txt",
    "The Jungle": "https://www.gutenberg.org/cache/epub/140/pg140.txt",
    "The Iron Heel": "https://www.gutenberg.org/cache/epub/1164/pg1164.txt"
}

POLES = {
    "agrarian": """
    Traditional agrarian life, seasonal crop reliance,
    rural harvesting, land ties, ancestral farming
    """,
    "collective": """
    Group harmony, sacrificing personal desires for
    family honor, community duty, filial piety, collective
    accountability
    """,
    "high_context": """
    Reading between the lines, heavily implied subtext,
    unspoken social hierarchies, indirect speech, saving face
    """
}

def get_embeddings(model: str, text: str | list[str]) -> NDArray:
    """Returns embeddings with L2 normalization"""
    
    result = ollama.embed(model, text).embeddings
    arr = np.array(result, dtype=np.float32)

    # Handle 1D arrays
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

#### Establish Vector Store ####

chunk_embeddings = []

for book, url in tqdm.tqdm(URLS.items()):
    text = requests.get(url).text.strip().split()
    chunk_range = range(0, len(text), CHUNK_WORDS)
    chunks = [" ".join(text[i : i + CHUNK_WORDS]) for i in chunk_range]
    chunk_embeddings.append(get_embeddings(EMBEDDING_MODEL, chunks))

chunk_embeddings = np.vstack(chunk_embeddings)

#### Generate Comparisons for Similarity ####

comparisons = []

for pole, description in tqdm.tqdm(POLES.items()):

    # Prompt for pole generation
    prompt = f"""
    You are an expert cross-cultural anthropologist and linguist.
    
    Generate exactly 10 distinct, highly descriptive phrases or
    micro-scenarios that represent an extreme baseline for this trait:
    {description}
    
    CRITICAL RULES:
    1. Isolate the concept. Make it pure, focused, and extreme.
    2. Keep them brief (8-15 words each).
    3. Return ONLY valid JSON matching the template below. 
    4. Do not include any reasoning, conversational text,
       markdown formatting blocks, or chatter. Only raw JSON.
    
    JSON Template:
    {{"phrases": ["phrase 1", "phrase 2", "phrase 3"]}}
    """
    
    # Responses from LLM with pole statements
    response = ollama.generate(
        model=LANGUAGE_MODEL,
        prompt=prompt,
        format="json",
        options={"temperature": 0.0, "seed": 42},
    )

    # Extract embeddings for pole statements
    phrases = json.loads(response["response"])["phrases"]
    pole_embeddings = get_embeddings(EMBEDDING_MODEL, phrases)
    similarity = np.dot(chunk_embeddings, pole_embeddings.T)

    # Store as dataframe for correlation analysis
    comparisons.append(
        pl.DataFrame(similarity)
        .with_row_index(
            name="id"
        )
        .unpivot(
            index="id",
            variable_name="phrase_id"
        )
        .group_by("id")
        .agg(
            pl.col("value").mean().alias(pole)
        )
    )

#### Visualize ####

joint_similarity = (
    reduce(
        lambda left, right: left.join(right, on="id"),
        comparisons
    )
    .select(
        "id",
        pl.col("agrarian").alias("Agrarian"),
        pl.col("collective").alias("Collective"),
        pl.col("high_context").alias("Context")
    )
    .unpivot(
        index="id"
    )
)

plot_df = (
    joint_similarity
    .join(
        other=joint_similarity,
        on="id"
    )
    .filter(
        pl.col("variable") < pl.col("variable_right")
    )
)

plot = (
    p9.ggplot(
        data=plot_df,
        mapping=p9.aes("value", "value_right")
    ) +
    p9.geom_point(
        size=3,
        color="white"
    ) +
    p9.geom_point(
        size=1
    ) +
    p9.geom_smooth(
        color="red"
    ) +
    p9.labs(
        title="Correlations between Literary Themes",
        subtitle="Using embedding similarity between select books",
        x="Cosine Similarity",
        y="Cosine Similarity"
    ) +
    p9.theme_minimal() +
    p9.theme(
        panel_grid_minor=p9.element_blank(),
        plot_title=p9.element_text(face="bold")
    ) +
    p9.facet_wrap(
        facets="~ variable + variable_right"
    )
)

plot.show()

print(
    plot_df
    .group_by("variable", "variable_right")
    .agg(
        pl.corr("value", "value_right").alias("correlation")
    )
)
```

## Resources

- [Gutenberg](https://www.gutenberg.org/)
- [Ollama](https://ollama.com/)
- [Google DeepMind](https://deepmind.google/)
- [Reading Scripture with Individualistic Eyes](https://www.ivpress.com/misreading-scripture-with-individualist-eyes)
