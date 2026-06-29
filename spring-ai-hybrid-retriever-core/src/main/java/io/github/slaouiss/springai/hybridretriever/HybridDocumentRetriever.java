package io.github.slaouiss.springai.hybridretriever;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.join.DocumentJoiner;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

/**
 * A {@link DocumentRetriever} that combines a dense (vector) retriever and a sparse (keyword)
 * retriever, then fuses their results.
 *
 * <p>The two retrievers are invoked sequentially; results are passed to a {@link
 * org.springframework.ai.rag.retrieval.join.DocumentJoiner} (default: {@link RRFDocumentJoiner
 * RRFDocumentJoiner with k=60}), and the top {@code topK} documents are returned. If either
 * retriever throws, the exception propagates unchanged — there is no silent fallback.
 *
 * <pre>{@code
 * DocumentRetriever hybrid = HybridDocumentRetriever.builder()
 *         .dense(vectorStoreRetriever)
 *         .sparse(luceneRetriever)
 *         .topK(10)
 *         .build();
 * }</pre>
 */
public final class HybridDocumentRetriever implements DocumentRetriever {

  private static final int DEFAULT_TOP_K = 10;

  private final DocumentRetriever dense;
  private final DocumentRetriever sparse;
  private final DocumentJoiner joiner;
  private final int topK;

  private HybridDocumentRetriever(
      DocumentRetriever dense, DocumentRetriever sparse, DocumentJoiner joiner, int topK) {
    this.dense = dense;
    this.sparse = sparse;
    this.joiner = joiner;
    this.topK = topK;
  }

  public static Builder builder() {
    return new Builder();
  }

  @Override
  public List<Document> retrieve(Query query) {
    Objects.requireNonNull(query, "query must not be null");
    List<Document> denseResults =
        Objects.requireNonNull(dense.retrieve(query), "dense retriever returned null");
    List<Document> sparseResults =
        Objects.requireNonNull(sparse.retrieve(query), "sparse retriever returned null");
    List<Document> joined =
        Objects.requireNonNull(
            joiner.join(Map.of(query, List.of(denseResults, sparseResults))),
            "joiner returned null");
    return List.copyOf(joined.subList(0, Math.min(topK, joined.size())));
  }

  public static final class Builder {

    private DocumentRetriever dense;
    private DocumentRetriever sparse;
    private DocumentJoiner joiner;
    private int topK = DEFAULT_TOP_K;

    private Builder() {}

    public Builder dense(DocumentRetriever dense) {
      this.dense = dense;
      return this;
    }

    public Builder sparse(DocumentRetriever sparse) {
      this.sparse = sparse;
      return this;
    }

    public Builder joiner(DocumentJoiner joiner) {
      this.joiner = Objects.requireNonNull(joiner, "joiner must not be null");
      return this;
    }

    public Builder topK(int topK) {
      if (topK <= 0) throw new IllegalArgumentException("topK must be positive, was: " + topK);
      this.topK = topK;
      return this;
    }

    public HybridDocumentRetriever build() {
      Objects.requireNonNull(dense, "dense must not be null");
      Objects.requireNonNull(sparse, "sparse must not be null");
      DocumentJoiner effectiveJoiner = joiner != null ? joiner : new RRFDocumentJoiner();
      return new HybridDocumentRetriever(dense, sparse, effectiveJoiner, topK);
    }
  }
}
