# Spring AI Hybrid Retriever

A composable hybrid `DocumentRetriever` for Spring AI — combine dense and sparse retrievers
into a single, reusable one, with Reciprocal Rank Fusion.

Spring AI provides the building blocks for modular RAG, but composing multiple retrieval
strategies into a single reusable `DocumentRetriever` still requires custom plumbing. This
library fills that gap.

- [The problem](#the-problem)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Using it with `RetrievalAugmentationAdvisor`](#using-it-with-retrievalaugmentationadvisor)
- [Configuration & extension points](#configuration--extension-points)
- [Spring Boot auto-configuration](#spring-boot-auto-configuration)
- [Why not native hybrid search?](#why-not-native-hybrid-search)
- [Scope and non-goals](#scope-and-non-goals-v1)
- [Requirements](#requirements)

---

## The problem

Spring AI ships the modular RAG SPI — `DocumentRetriever`, `DocumentJoiner`,
`RetrievalAugmentationAdvisor` — and a `VectorStoreDocumentRetriever` for dense (semantic) search.
What it does **not** ship is a ready-made way to run a dense retriever *and* a sparse (keyword)
retriever and fuse their results: the only built-in `DocumentJoiner`,
`ConcatenationDocumentJoiner`, simply concatenates and keeps the first occurrence of a duplicate.

So today every team rewrites the same plumbing:

```java
List<Document> dense  = denseRetriever.retrieve(query);
List<Document> sparse = sparseRetriever.retrieve(query);
List<Document> merged = /* fuse + dedup by hand */;
```

`spring-ai-hybrid-retriever` turns that into a single component that implements Spring AI's own
`DocumentRetriever` interface, so it drops straight into the modular RAG pipeline.

> **Scope.** This library does *not* reimplement keyword search — dense and sparse search are
> delegated to retrievers you provide. Its job is **composition and fusion** at the application
> layer. (See [Why not native hybrid search?](#why-not-native-hybrid-search) below.)

---

## Installation

Add the starter (Spring Boot auto-configuration) or the core module (no Spring Boot dependency).

```xml
<!-- Spring Boot apps: auto-configuration included -->
<dependency>
    <groupId>io.github.slaouiss</groupId>
    <artifactId>spring-ai-hybrid-retriever-spring-boot-starter</artifactId>
    <version>1.0.0</version>
</dependency>
```

```xml
<!-- Or core only, framework-agnostic -->
<dependency>
    <groupId>io.github.slaouiss</groupId>
    <artifactId>spring-ai-hybrid-retriever-core</artifactId>
    <version>1.0.0</version>
</dependency>
```

---

## Quick start

`dense` and `sparse` are **any** Spring AI `DocumentRetriever`. The dense one is typically a
`VectorStoreDocumentRetriever`; the sparse one is any retriever backed by a keyword/full-text
engine (Lucene, Elasticsearch, OpenSearch, PostgreSQL FTS, …).

```java
// Dense: any Spring AI DocumentRetriever — typically the built-in vector-store one.
DocumentRetriever dense = VectorStoreDocumentRetriever.builder()
        .vectorStore(vectorStore)
        .topK(20)
        .build();

// Sparse: your own DocumentRetriever over a keyword/full-text engine.
// (Implementing one is a few lines — see the `demo` module for a Lucene-backed example.)
DocumentRetriever sparse = new MyLuceneDocumentRetriever(luceneIndex, 20);

DocumentRetriever hybrid = HybridDocumentRetriever.builder()
        .dense(dense)
        .sparse(sparse)
        .topK(10)            // number of fused results to return
        .build();            // Reciprocal Rank Fusion is the default joiner

List<Document> results = hybrid.retrieve(new Query("how do I rotate my API keys?"));
```

That's the whole API surface for the common case. RRF (`k = 60`) and identity-by-id are the
defaults; both are configurable (see below).

---

## How it works

```
                         Query
                           │
                 HybridDocumentRetriever
                  /                      \
        DocumentRetriever          DocumentRetriever
            (dense)                    (sparse)
                  \                      /
                   RRFDocumentJoiner
                  (+ DocumentIdentityStrategy)
                           │
                     List<Document>   (ranked, top-K)
```

1. Run the dense and the sparse retriever.
2. Fuse both ranked lists with **Reciprocal Rank Fusion**: each document scores
   `Σ 1 / (k + rank)` over every list it appears in (ranks are 1-based). RRF works on ranks, not
   scores, so it sidesteps the fact that semantic-similarity and keyword scores live on
   incompatible scales.
3. **Documents that appear in *both* lists have their contributions summed** — via a
   `DocumentIdentityStrategy` (by id, by default). That overlap is where fusion adds value, and it
   is exactly what the built-in `ConcatenationDocumentJoiner` discards.
4. Sort, truncate to `topK`.

---

## Using it with `RetrievalAugmentationAdvisor`

Because `HybridDocumentRetriever` is a `DocumentRetriever`, it plugs into Spring AI's modular RAG
advisor with no other changes:

```java
Advisor rag = RetrievalAugmentationAdvisor.builder()
        .documentRetriever(hybrid)
        .build();

String answer = chatClient.prompt()
        .advisors(rag)
        .user(question)
        .call()
        .content();
```

---

## Configuration & extension points

Everything beyond the defaults is an explicit, replaceable component:

| Concern | Default | How to change |
|---|---|---|
| Fusion algorithm | `RRFDocumentJoiner` (k = 60) | `.joiner(new RRFDocumentJoiner(50))`, or any `DocumentJoiner` |
| RRF constant `k` | 60 | constructor argument of `RRFDocumentJoiner` |
| Fused result count | 10 | `.topK(n)` |
| Document identity | `ByIdDocumentIdentityStrategy` (`document.getId()`) | pass a `DocumentIdentityStrategy` to the joiner |

```java
// Custom identity: merge documents that share a business key across sources.
DocumentIdentityStrategy byBusinessKey =
        document -> document.getMetadata().get("sourceId");

DocumentRetriever hybrid = HybridDocumentRetriever.builder()
        .dense(dense)
        .sparse(sparse)
        .joiner(new RRFDocumentJoiner(60, byBusinessKey))
        .topK(10)
        .build();
```

`RRFDocumentJoiner` is a standalone `DocumentJoiner` — it has no dependency on
`HybridDocumentRetriever` and can be used on its own anywhere a `DocumentJoiner` is expected.

---

## Spring Boot auto-configuration

With the starter on the classpath, a `HybridDocumentRetriever` bean is auto-configured when a
dense retriever and a sparse retriever are available. It backs off (`@ConditionalOnMissingBean`)
if you define your own.

```properties
spring.ai.hybrid-retriever.rrf-k=60     # RRF smoothing constant (default 60)
spring.ai.hybrid-retriever.top-k=10     # fused results returned (default 10)
```

The dense retriever can be auto-wired from a `VectorStore` bean. The sparse retriever is always
supplied by you, since the library deliberately does not bind to any specific keyword engine.

---

## Why not native hybrid search?

Many engines (Elasticsearch, OpenSearch, Qdrant, Oracle, …) already fuse dense and sparse search
natively, in a single query. When your dense and sparse indexes live in the **same** engine, use
its native hybrid search — it will be faster than fusing in the application.

This library is for the cases the native path can't cover:

- your dense and sparse search live in **different** systems (e.g. a vector store plus a separate
  keyword engine), which no single-store hybrid can fuse;
- you want fusion to stay a plain Spring AI `DocumentRetriever` so you can swap stores without
  coupling your code to a store-specific hybrid API;
- you want to control or replace the fusion strategy yourself.

It composes and fuses; it does not reimplement search.

---

## Scope and non-goals (V1)

Kept deliberately small. The following are **out of scope for now**, by design:

- Parallel execution of the dense/sparse retrievers (V1 runs them sequentially; if a retriever
  fails, the call fails — no silent fallback).
- Timeouts, retries, circuit breaking.
- Weighted RRF and alternative fusion strategies (the `DocumentJoiner` SPI lets you add your own).
- Cross-encoder reranking, chunking, data-source connectors.

These are intentionally left to dedicated components rather than bolted on here.

---

## Requirements

- Java 17+
- Spring AI 1.1+

---

## License

Licensed under the Apache License 2.0.

---

*Designed as a natural extension of Spring AI's modular RAG architecture — it adds a retrieval
strategy, it does not replace any part of the pipeline.*