package io.github.slaouiss.springai.hybridretriever.demo;

import io.github.slaouiss.springai.hybridretriever.HybridDocumentRetriever;
import java.util.ArrayList;
import java.util.List;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformers.TransformersEmbeddingModel;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;

public class HybridRetrieverDemo {

  // Pinned to commit eb3992ca — the v1.1.8 release tag. Immutable.
  private static final String MODEL_URL =
      "https://media.githubusercontent.com/media/spring-projects/spring-ai/"
          + "eb3992ca27a9931ca79d7e2a817badfbb0fa6867"
          + "/models/spring-ai-transformers/src/main/resources/onnx/all-MiniLM-L6-v2/model.onnx";

  // Tokenizer bundled inside spring-ai-transformers.jar — no download required.
  private static final String TOKENIZER_URI = "classpath:onnx/all-MiniLM-L6-v2/tokenizer.json";

  private static final String CACHE_DIR =
      System.getProperty("user.home") + "/.cache/spring-ai-onnx";

  private static final int TOP_K = 5;

  private static final List<String> QUERIES =
      List.of(
          "Spring AI DocumentRetriever vector store pipeline",
          "BM25 term frequency inverted index Lucene StandardAnalyzer",
          "HNSW FAISS approximate nearest neighbor vector index recall",
          "retrieval augmented generation knowledge grounding evidence");

  public static void main(String[] args) throws Exception {
    var documents = loadDocuments();
    System.out.printf("Loaded %d documents.%n%n", documents.size());

    System.out.println("Initialising embedding model (downloads ~86 MB on first run)...");
    var embeddingModel = buildEmbeddingModel();
    System.out.println("Ready.\n");

    var vectorStore = SimpleVectorStore.builder(embeddingModel).build();
    vectorStore.add(documents);
    var denseRetriever =
        VectorStoreDocumentRetriever.builder().vectorStore(vectorStore).topK(TOP_K).build();

    var sparseRetriever = new LuceneDocumentRetriever(documents, TOP_K);

    var hybridRetriever =
        HybridDocumentRetriever.builder()
            .dense(denseRetriever)
            .sparse(sparseRetriever)
            .topK(TOP_K)
            .build();

    for (var queryText : QUERIES) {
      runQuery(queryText, denseRetriever, sparseRetriever, hybridRetriever);
    }
  }

  // ── Setup ──────────────────────────────────────────────────────────────────

  private static List<Document> loadDocuments() throws Exception {
    var resolver = new PathMatchingResourcePatternResolver();
    var resources = resolver.getResources("classpath:documents/*.txt");
    var documents = new ArrayList<Document>();
    for (var resource : resources) {
      documents.addAll(new TextReader(resource).get());
    }
    return documents;
  }

  private static TransformersEmbeddingModel buildEmbeddingModel() throws Exception {
    var model = new TransformersEmbeddingModel();
    model.setTokenizerResource(TOKENIZER_URI);
    model.setModelResource(MODEL_URL);
    model.setResourceCacheDirectory(CACHE_DIR);
    model.afterPropertiesSet();
    return model;
  }

  // ── Query execution ────────────────────────────────────────────────────────

  private static void runQuery(
      String queryText,
      DocumentRetriever dense,
      DocumentRetriever sparse,
      DocumentRetriever hybrid) {
    var query = new Query(queryText);
    var line = "=".repeat(49);
    System.out.println(line);
    System.out.println("QUERY");
    System.out.println(line);
    System.out.println();
    System.out.println(queryText);
    System.out.println();
    printList("Dense", dense.retrieve(query));
    printList("Sparse", sparse.retrieve(query));
    printList("Hybrid (RRF)", hybrid.retrieve(query));
  }

  private static void printList(String label, List<Document> docs) {
    var line = "-".repeat(49);
    System.out.println(line);
    System.out.println(label);
    System.out.println(line);
    System.out.println();
    for (int i = 0; i < docs.size(); i++) {
      var doc = docs.get(i);
      var score = doc.getScore();
      var scoreStr = score != null ? String.format("(%.5f)", score) : "(n/a)";
      System.out.printf("  %d. %-44s %s%n", i + 1, sourceName(doc), scoreStr);
    }
    if (docs.isEmpty()) System.out.println("  (no results)");
    System.out.println();
  }

  private static String sourceName(Document doc) {
    var source = String.valueOf(doc.getMetadata().getOrDefault("source", doc.getId()));
    var slash = source.lastIndexOf('/');
    return slash >= 0 ? source.substring(slash + 1) : source;
  }
}
