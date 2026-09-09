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

# IMPORTANT: This assumes a normalized embedding model!
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
    "industrial": """
    Industrial automation, assembly line labor, urban
    manufacturing, hourly wages, corporate clocking-in
    """,
    "collective": """
    Group harmony, sacrificing personal desires for
    family honor, community duty, filial piety, collective
    accountability
    """,
    "individual": """
    Radical self-reliance, individual autonomy, personal
    freedom, pursuing independent ambition, breaking social
    conformity
    """,
    "high_context": """
    Reading between the lines, heavily implied subtext,
    unspoken social hierarchies, indirect speech, saving face
    """,
    "low_context": """
    Direct and literal speech, explicit legal contracts,
    spelling everything out plainly, unambiguous statements
    """
}

def get_embeddings(model: str, text: str | list[str]) -> NDArray:
    result = ollama.embed(model, text).embeddings
    return np.array(result, dtype=np.float32)

#### Establish Vector Store ####

chunk_embeddings = []

for book, url in tqdm.tqdm(URLS.items()):
    text = requests.get(url).text.strip().split(" ")
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
        (pl.col("agrarian") - pl.col("industrial")).alias("Agrarian"),
        (pl.col("collective") - pl.col("individual")).alias("Collective"),
        (pl.col("low_context") - pl.col("high_context")).alias("Context")
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
    p9.facet_grid(
        cols="variable",
        rows="variable_right"
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
