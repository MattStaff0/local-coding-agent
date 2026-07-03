---
url: https://docs.trychroma.com/docs/querying-collections/query-and-get
fetched: 2026-07-02
---

The Query API enables nearest-neighbor similarity search over dense embeddings.
Use the Get API when you want to retrieve records without similarity ranking.

* Python
* TypeScript
* Rust

## Query

You can query a collection to run a similarity search using `.query`:

```
collection.query(
    query_texts=["thus spake zarathustra", "the oracle speaks"]
)
```

Chroma will use the collection’s [embedding function](../embeddings/embedding-functions) to embed your text queries, and use the output to run a vector similarity search against your collection.Instead of providing `query_texts`, you can provide `query_embeddings` directly. You will be required to do so if your collection does not have an embedding function attached to it. The dimension of your query embedding must match the dimension of the embeddings in your collection.Python also supports `query_images` and `query_uris` as query inputs.

```
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1], [1.1, 2.3, 3.2]]
)
```

By default, Chroma will return 10 results per input query. You can modify this number using the `n_results` argument:

```
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1], [1.1, 2.3, 3.2]],
    n_results=100
)
```

The `ids` argument lets you constrain the search only to records with the IDs from the provided list:

```
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1], [1.1, 2.3, 3.2]],
    n_results=100,
    ids=["id1", "id2"]
)
```

Both `query` and `get` support `where` for [metadata filtering](./metadata-filtering) and `where_document` for [full-text search and regex](./full-text-search):

```
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1], [1.1, 2.3, 3.2]],
    n_results=100,
    where={"page": 10}, # query records with metadata field 'page' equal to 10
    where_document={"$contains": "search string"} # query records with the search string in the records' document
)
```

## Get

Use `.get` to retrieve records by ID and/or filters without similarity ranking:

```
collection.get(ids=["id1", "id2"]) # by IDs

collection.get(limit=100, offset=0) # with pagination
```

## Query

You can query a collection to run a similarity search using `.query`:

```
await collection.query({
  queryTexts: ["thus spake zarathustra", "the oracle speaks"],
});
```

Chroma will use the collection’s [embedding function](../embeddings/embedding-functions) to embed your text queries, and use the output to run a vector similarity search against your collection.Instead of providing `queryTexts`, you can provide `queryEmbeddings` directly. You will be required to do so if your collection does not have an embedding function attached to it. The dimension of your query embedding must match the dimension of the embeddings in your collection.

```
await collection.query({
  queryEmbeddings: [
    [11.1, 12.1, 13.1],
    [1.1, 2.3, 3.2],
  ],
});
```

By default, Chroma will return 10 results per input query. You can modify this number using the `nResults` argument:

```
await collection.query({
  queryEmbeddings: [
    [11.1, 12.1, 13.1],
    [1.1, 2.3, 3.2],
  ],
  nResults: 100,
});
```

The `ids` argument lets you constrain the search only to records with the IDs from the provided list:

```
await collection.query({
  queryEmbeddings: [
    [11.1, 12.1, 13.1],
    [1.1, 2.3, 3.2],
  ],
  nResults: 100,
  ids: ["id1", "id2"],
});
```

Both `query` and `get` support `where` for [metadata filtering](./metadata-filtering) and `whereDocument` for [full-text search and regex](./full-text-search):

```
await collection.query({
  queryEmbeddings: [
    [11.1, 12.1, 13.1],
    [1.1, 2.3, 3.2],
  ],
  nResults: 5,
  where: { page: 10 }, // metadata field 'page' equal to 10
  whereDocument: { $contains: "search string" }, // documents containing "search string"
});
```

## Get

Use `.get` to retrieve records by ID and/or filters without similarity ranking:

```
await collection.get({ ids: ["id1", "id2"] }); // By IDs

await collection.get({ limit: 100, offset: 0 }); // With pagination
```

## Type inference

You can also pass type arguments to `.get` and `.query` for the shape of your metadata. This gives you type inference for your metadata objects:

```
const results = await collection.get<{page: number; title: string}>({
  ids: ["id1", "id2"],
});

const rows = results.rows();
rows.forEach((row) => {
  console.log(row.id, row.metadata?.page);
});
```

## Query

You can query a collection to run a similarity search using `.query`:

```
use chroma_types::IncludeList;

// pub async fn query(
//    &self,
//    query_embeddings: Vec<Vec<f32>>,
//    n_results: Option<u32>,
//    where: Option<Where>,
//    ids: Option<Vec<String>>,
//    include: Option<IncludeList>,
// ) -> Result<QueryResponse, ChromaHttpClientError>

let results = collection
    .query(
        vec![vec![11.1, 12.1, 13.1], vec![1.1, 2.3, 3.2]],
        None,
        None,
        None,
        None,
    )
    .await?;
```

Embeddings must be provided directly to the Rust client.By default, Chroma returns 10 results per input query. You can modify this number using `n_results`:

```
let results = collection
    .query(
        vec![vec![11.1, 12.1, 13.1], vec![1.1, 2.3, 3.2]],
        Some(100), // n_results
        None,
        None,
        None,
    )
    .await?;
```

The `ids` argument lets you constrain the search only to records with the IDs from the provided list:

```
let results = collection
    .query(
        vec![vec![11.1, 12.1, 13.1], vec![1.1, 2.3, 3.2]],
        Some(5),
        None,
        Some(vec!["id1".to_string(), "id2".to_string()]), // ids
        None,
    )
    .await?;
```

## Get

Use `.get` to retrieve records by ID and/or filters without similarity ranking:

```
let response = collection
    .get(
        Some(vec!["id1".to_string(), "id2".to_string()]),
        None,
        Some(10),
        Some(0),
        Some(IncludeList::default_get()),
    )
    .await?;
```

## Results Shape

Chroma returns `.query` and `.get` results in **column-major** form (arrays per field). `.query` results are grouped per input query; `.get` results are a flat list of records.

Python

TypeScript

Rust

```
class QueryResult(TypedDict):
    ids: List[IDs]
    embeddings: Optional[List[Embeddings]]
    documents: Optional[List[List[Document]]]
    uris: Optional[List[List[URI]]]
    metadatas: Optional[List[List[Metadata]]]
    distances: Optional[List[List[float]]]
    included: Include

class GetResult(TypedDict):
    ids: List[ID]
    embeddings: Optional[Embeddings]
    documents: Optional[List[Document]]
    uris: Optional[URIs]
    metadatas: Optional[List[Metadata]]
    included: Include
```

Here is a concrete example of what these responses look like in practice:

```
// Query result
{
  "ids": [["doc_1", "doc_7"]],
  "embeddings": [[[1, 2, 3, 4], [1, 2, 3, 4]]],
  "documents": [["Chroma stores vectors.", "Embeddings power semantic search."]],
  "metadatas": [[
    {"source": "docs", "topic": "intro"},
    {"source": "blog", "topic": "search"}
  ]],
  "distances": [[0.12, 0.21]],
  "included": ["embeddings", "documents", "metadatas", "distances"]
}
// Get result
{
  "ids": ["doc_1", "doc_7"],
  "embeddings": [[1, 2, 3, 4], [1, 2, 3, 4]],
  "documents": ["Chroma stores vectors.", "Embeddings power semantic search."],
  "metadatas": [
    {"source": "docs", "topic": "intro"},
    {"source": "blog", "topic": "search"}
  ],
  "included": ["documents", "metadatas"]
}
```

In the results from the Get operation, corresponding elements in each array belong
to the same document.

Python

TypeScript

Rust

```
result = collection.get(include=["documents", "metadatas"])
for id, document, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
    print(id, document, metadata)
```

Query is a batch API and returns results grouped per input. A common pattern is to iterate over each query’s “batch” of results, then iterate within that batch.

Python

TypeScript

Rust

```
result = collection.query(query_texts=["first query", "second query"])
for ids, documents, metadatas in zip(result["ids"], result["documents"], result["metadatas"]):
    for id, document, metadata in zip(ids, documents, metadatas):
        print(id, document, metadata)
```

## Choosing Which Data is Returned

By default, Query returns `documents`, `metadatas`, and `distances`, and Get returns `documents` and `metadatas`.
Use `include` to control what comes back. `ids` are always returned.

Python

TypeScript

Rust

```
collection.query(
    query_texts=["my query"],
    include=["documents", "metadatas", "embeddings"],
)

collection.get(include=["documents"])
```

Was this page helpful?

YesNo

[Suggest edits](https://github.com/chroma-core/chroma/edit/main/docs/mintlify/docs/querying-collections/query-and-get.mdx)

[Configure Collections

Previous](/docs/collections/configure)[Metadata Filtering

Next](/docs/querying-collections/metadata-filtering)

⌘I
