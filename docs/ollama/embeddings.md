---
url: https://docs.ollama.com/capabilities/embeddings
fetched: 2026-07-02
---

[Guide](/)[Integrations](/integrations)[API Reference](/api/introduction)

Embeddings turn text into numeric vectors you can store in a vector database, search with cosine similarity, or use in RAG pipelines. The vector length depends on the model (typically 384–1024 dimensions).

## Recommended models

* [embeddinggemma](https://ollama.com/library/embeddinggemma)
* [qwen3-embedding](https://ollama.com/library/qwen3-embedding)
* [all-minilm](https://ollama.com/library/all-minilm)

## Generate embeddings

* CLI
* cURL
* Python
* JavaScript

Generate embeddings directly from the command line:

```
ollama run embeddinggemma "Hello world"
```

You can also pipe text to generate embeddings:

```
echo "Hello world" | ollama run embeddinggemma
```

Output is a JSON array.

```
curl -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma",
    "input": "The quick brown fox jumps over the lazy dog."
  }'
```

```
import ollama

single = ollama.embed(
  model='embeddinggemma',
  input='The quick brown fox jumps over the lazy dog.'
)
print(len(single['embeddings'][0]))  # vector length
```

```
import ollama from 'ollama'

const single = await ollama.embed({
  model: 'embeddinggemma',
  input: 'The quick brown fox jumps over the lazy dog.',
})
console.log(single.embeddings[0].length) // vector length
```

The `/api/embed` endpoint returns L2‑normalized (unit‑length) vectors.

## Generate a batch of embeddings

Pass an array of strings to `input`.

* cURL
* Python
* JavaScript

```
curl -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma",
    "input": [
      "First sentence",
      "Second sentence",
      "Third sentence"
    ]
  }'
```

```
import ollama

batch = ollama.embed(
  model='embeddinggemma',
  input=[
    'The quick brown fox jumps over the lazy dog.',
    'The five boxing wizards jump quickly.',
    'Jackdaws love my big sphinx of quartz.',
  ]
)
print(len(batch['embeddings']))  # number of vectors
```

```
import ollama from 'ollama'

const batch = await ollama.embed({
  model: 'embeddinggemma',
  input: [
    'The quick brown fox jumps over the lazy dog.',
    'The five boxing wizards jump quickly.',
    'Jackdaws love my big sphinx of quartz.',
  ],
})
console.log(batch.embeddings.length) // number of vectors
```

## Tips

* Use cosine similarity for most semantic search use cases.
* Use the same embedding model for both indexing and querying.

[Previous](/capabilities/vision)[Tool calling

Next](/capabilities/tool-calling)

⌘I
