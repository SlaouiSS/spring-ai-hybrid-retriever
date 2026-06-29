package io.github.slaouiss.springai.hybridretriever.benchmark;

import io.github.slaouiss.springai.hybridretriever.HybridDocumentRetriever;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

/**
 * Measures HybridDocumentRetriever.retrieve() orchestration overhead.
 *
 * <p>Both the dense and sparse retrievers are stubs ({@link FixedListRetriever}) that return a
 * pre-built list immediately, with no IO, no search, and no embedding generation. The measured cost
 * is therefore: two stub calls + Map.of + List.of + join() + subList + List.copyOf.
 *
 * <p>topK is set to 2 * listSize so the subList/copyOf path never truncates, keeping this benchmark
 * comparable to {@link RrfDocumentJoinerBenchmark}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class HybridDocumentRetrieverBenchmark {

  @Param({"10", "50", "100", "500", "1000"})
  public int listSize;

  private HybridDocumentRetriever retriever;
  private Query query;

  @Setup(Level.Trial)
  public void setUp() {
    query = new Query("benchmark query");
    List<Document> dense = buildDocumentList(0, listSize);
    List<Document> sparse = buildDocumentList(listSize / 2, listSize);
    retriever =
        HybridDocumentRetriever.builder()
            .dense(new FixedListRetriever(dense))
            .sparse(new FixedListRetriever(sparse))
            .topK(listSize * 2)
            .build();
  }

  @Benchmark
  public List<Document> retrieve() {
    return retriever.retrieve(query);
  }

  private static List<Document> buildDocumentList(int startId, int count) {
    List<Document> docs = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      String id = String.valueOf(startId + i);
      docs.add(Document.builder().id(id).text(id).build());
    }
    return docs;
  }

  private static final class FixedListRetriever implements DocumentRetriever {

    private final List<Document> documents;

    FixedListRetriever(List<Document> documents) {
      this.documents = documents;
    }

    @Override
    public List<Document> retrieve(Query query) {
      return documents;
    }
  }
}
