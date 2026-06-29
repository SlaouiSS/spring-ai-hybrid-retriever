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

/**
 * Measures RRFDocumentJoiner.join() in isolation.
 *
 * <p>Input: two ranked lists of {@code listSize} documents with 50% overlap (documents
 * listSize/2..listSize-1 appear in both the dense and sparse lists). This exercises the merge path
 * — the most interesting case for RRF.
 *
 * <p>Nothing except the join algorithm is on the hot path: document lists are pre-built in
 * {@code @Setup} and reused across all iterations.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class RrfDocumentJoinerBenchmark {

  @Param({"10", "50", "100", "500", "1000"})
  public int listSize;

  private RRFDocumentJoiner joiner;
  private Map<Query, List<List<Document>>> input;

  @Setup(Level.Trial)
  public void setUp() {
    joiner = new RRFDocumentJoiner();
    Query query = new Query("benchmark query");
    // dense: ids 0 .. listSize-1
    // sparse: ids listSize/2 .. (3*listSize/2)-1  →  50% overlap with dense
    List<Document> dense = buildDocumentList(0, listSize);
    List<Document> sparse = buildDocumentList(listSize / 2, listSize);
    input = Map.of(query, List.of(dense, sparse));
  }

  @Benchmark
  public List<Document> join() {
    return joiner.join(input);
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
