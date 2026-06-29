package io.github.slaouiss.springai.hybridretriever.benchmark;

import io.github.slaouiss.springai.hybridretriever.RRFDocumentJoiner;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
import org.springframework.ai.rag.retrieval.join.ConcatenationDocumentJoiner;

/**
 * Compares ConcatenationDocumentJoiner vs RRFDocumentJoiner on identical inputs.
 *
 * <p>Both benchmark methods receive the same pre-built input map, so any measured difference
 * reflects the additional cost of the RRF algorithm (rank tracking, score accumulation, sorting)
 * over a simple concatenation.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class JoinerComparisonBenchmark {

  @Param({"10", "50", "100", "500", "1000"})
  public int listSize;

  private RRFDocumentJoiner rrfJoiner;
  private ConcatenationDocumentJoiner concatenationJoiner;
  private Map<Query, List<List<Document>>> input;

  @Setup(Level.Trial)
  public void setUp() {
    rrfJoiner = new RRFDocumentJoiner();
    concatenationJoiner = new ConcatenationDocumentJoiner();
    Query query = new Query("benchmark query");
    List<Document> dense = buildDocumentList(0, listSize);
    List<Document> sparse = buildDocumentList(listSize / 2, listSize);
    input = Map.of(query, List.of(dense, sparse));
  }

  @Benchmark
  public List<Document> rrfJoin() {
    return rrfJoiner.join(input);
  }

  @Benchmark
  public List<Document> concatenationJoin() {
    return concatenationJoiner.join(input);
  }

  private static List<Document> buildDocumentList(int startId, int count) {
    List<Document> docs = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      String id = String.valueOf(startId + i);
      docs.add(Document.builder().id(id).text(id).build());
    }
    return docs;
  }
}
