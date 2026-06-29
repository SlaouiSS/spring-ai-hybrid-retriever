package io.github.slaouiss.springai.hybridretriever;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.join.DocumentJoiner;

/**
 * {@link DocumentJoiner} implementation using Reciprocal Rank Fusion (Cormack et al., 2009).
 *
 * <p>Each document scores {@code 1 / (k + rank)} per ranked list it appears in; contributions are
 * summed across all lists. Ranks are 1-based: the top document has rank 1, scoring {@code 1 / (k +
 * 1)}. A document present in both the dense and sparse list has two terms summed — that
 * cross-source overlap is where fusion adds value over simple concatenation.
 *
 * <p>Constructor options:
 *
 * <pre>{@code
 * new RRFDocumentJoiner()                    // k = 60, identity by document id
 * new RRFDocumentJoiner(50)                  // custom k, identity by document id
 * new RRFDocumentJoiner(byBusinessKey)       // k = 60, custom identity strategy
 * new RRFDocumentJoiner(50, byBusinessKey)   // custom k and identity strategy
 * }</pre>
 *
 * <p>This class has no dependency on {@link HybridDocumentRetriever} and can be used standalone
 * anywhere a {@code DocumentJoiner} is needed.
 */
public final class RRFDocumentJoiner implements DocumentJoiner {

  private static final int DEFAULT_K = 60;

  private final int k;
  private final DocumentIdentityStrategy identityStrategy;

  public RRFDocumentJoiner() {
    this(DEFAULT_K, new ByIdDocumentIdentityStrategy());
  }

  public RRFDocumentJoiner(int k) {
    this(k, new ByIdDocumentIdentityStrategy());
  }

  public RRFDocumentJoiner(DocumentIdentityStrategy identityStrategy) {
    this(DEFAULT_K, identityStrategy);
  }

  public RRFDocumentJoiner(int k, DocumentIdentityStrategy identityStrategy) {
    if (k <= 0) throw new IllegalArgumentException("k must be positive, was: " + k);
    Objects.requireNonNull(identityStrategy, "identityStrategy must not be null");
    this.k = k;
    this.identityStrategy = identityStrategy;
  }

  @Override
  public List<Document> join(Map<Query, List<List<Document>>> documentsForQuery) {
    Objects.requireNonNull(documentsForQuery, "documentsForQuery must not be null");
    Map<Object, Candidate> candidates = scan(documentsForQuery);
    return computeScores(candidates);
  }

  private Map<Object, Candidate> scan(Map<Query, List<List<Document>>> documentsForQuery) {
    Map<Object, Candidate> candidates = new LinkedHashMap<>();
    int sourceIndex = 0;
    for (List<List<Document>> rankedLists : documentsForQuery.values()) {
      for (List<Document> rankedList : rankedLists) {
        for (int i = 0; i < rankedList.size(); i++) {
          Document doc = rankedList.get(i);
          int rank = i + 1;
          Object identity = identityStrategy.identity(doc);
          if (identity == null)
            throw new IllegalArgumentException(
                "DocumentIdentityStrategy returned null for document id=" + doc.getId());
          candidates
              .computeIfAbsent(identity, key -> new Candidate(doc))
              .recordBestRank(sourceIndex, rank);
        }
        sourceIndex++;
      }
    }
    return candidates;
  }

  private List<Document> computeScores(Map<Object, Candidate> candidates) {
    return candidates.values().stream()
        .map(c -> c.representative.mutate().score(c.rrfScore(k)).build())
        .sorted((d1, d2) -> Double.compare(d2.getScore(), d1.getScore()))
        .collect(Collectors.toList());
  }

  private static final class Candidate {
    final Document representative;
    private final Map<Integer, Integer> bestRankPerSource = new LinkedHashMap<>();

    Candidate(Document representative) {
      this.representative = representative;
    }

    void recordBestRank(int sourceIndex, int rank) {
      bestRankPerSource.merge(sourceIndex, rank, Math::min);
    }

    double rrfScore(int k) {
      return bestRankPerSource.values().stream().mapToDouble(rank -> 1.0 / ((long) k + rank)).sum();
    }
  }
}
