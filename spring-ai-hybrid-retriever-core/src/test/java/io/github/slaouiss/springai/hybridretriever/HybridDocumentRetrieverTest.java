package io.github.slaouiss.springai.hybridretriever;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.join.DocumentJoiner;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

@ExtendWith(MockitoExtension.class)
class HybridDocumentRetrieverTest {

  @Mock DocumentRetriever dense;
  @Mock DocumentRetriever sparse;
  @Mock DocumentJoiner joiner;

  private final Query query = new Query("test query");

  private static Document doc(String id) {
    return Document.builder().id(id).text(id).build();
  }

  private static List<Document> docs(int count) {
    return IntStream.range(0, count).mapToObj(i -> doc("d" + i)).collect(Collectors.toList());
  }

  // -------------------------------------------------------------------------
  // 1. Builder validation
  // -------------------------------------------------------------------------

  @Test
  void builder_missingDense_throwsNullPointerException() {
    assertThatThrownBy(
            () -> HybridDocumentRetriever.builder().sparse(sparse).joiner(joiner).build())
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("dense");
  }

  @Test
  void builder_missingSparse_throwsNullPointerException() {
    assertThatThrownBy(() -> HybridDocumentRetriever.builder().dense(dense).joiner(joiner).build())
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("sparse");
  }

  @Test
  void builder_nullJoiner_throwsNullPointerException() {
    assertThatThrownBy(
            () ->
                HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(null).build())
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("joiner");
  }

  @Test
  void builder_zeroTopK_throwsIllegalArgumentException() {
    assertThatThrownBy(
            () ->
                HybridDocumentRetriever.builder()
                    .dense(dense)
                    .sparse(sparse)
                    .joiner(joiner)
                    .topK(0)
                    .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("topK");
  }

  @Test
  void builder_negativeTopK_throwsIllegalArgumentException() {
    assertThatThrownBy(
            () ->
                HybridDocumentRetriever.builder()
                    .dense(dense)
                    .sparse(sparse)
                    .joiner(joiner)
                    .topK(-1)
                    .build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void builder_defaultJoiner_isRRFDocumentJoiner() {
    // No explicit joiner → default is RRFDocumentJoiner with k=60.
    // Each document is at rank 1 in its own list → RRF score = 1/(60+1).
    DocumentRetriever denseRetriever = q -> List.of(doc("a"));
    DocumentRetriever sparseRetriever = q -> List.of(doc("b"));

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(denseRetriever).sparse(sparseRetriever).build();

    List<Document> result = retriever.retrieve(query);

    assertThat(result).hasSize(2);
    assertThat(result.get(0).getScore()).isCloseTo(1.0 / 61, within(1e-10));
    assertThat(result.get(1).getScore()).isCloseTo(1.0 / 61, within(1e-10));
  }

  @Test
  void builder_defaultTopK_isTen() {
    // Joiner returns 12 docs; the default topK must limit the result to exactly 10.
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(docs(12));

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    assertThat(retriever.retrieve(query)).hasSize(10);
  }

  @Test
  void builder_defaultIdentityStrategy_isByIdDocumentIdentityStrategy() {
    // Both sources return a document with the same ID.
    // The default ByIdDocumentIdentityStrategy must merge them into one.
    String sharedId = "article-1";
    DocumentRetriever denseRetriever =
        q -> List.of(Document.builder().id(sharedId).text("dense version").build());
    DocumentRetriever sparseRetriever =
        q -> List.of(Document.builder().id(sharedId).text("sparse version").build());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(denseRetriever).sparse(sparseRetriever).build();

    List<Document> result = retriever.retrieve(query);

    assertThat(result).hasSize(1);
    assertThat(result.get(0).getId()).isEqualTo(sharedId);
  }

  @Test
  void builder_customJoiner_isPreserved() {
    // Passing a custom joiner → that joiner (not the default RRFDocumentJoiner) is called.
    when(dense.retrieve(query)).thenReturn(List.of(doc("x")));
    when(sparse.retrieve(query)).thenReturn(List.of(doc("y")));
    when(joiner.join(any())).thenReturn(List.of(doc("from-custom-joiner")));

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    retriever.retrieve(query);

    verify(joiner).join(any());
  }

  @Test
  void builder_customTopK_isPreserved() {
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(docs(5));

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder()
            .dense(dense)
            .sparse(sparse)
            .joiner(joiner)
            .topK(3)
            .build();

    assertThat(retriever.retrieve(query)).hasSize(3);
  }

  // -------------------------------------------------------------------------
  // 2. Retrieval orchestration
  // -------------------------------------------------------------------------

  @Test
  void retrieve_denseRetriever_invokedExactlyOnce() {
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(List.of());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    retriever.retrieve(query);

    verify(dense, times(1)).retrieve(query);
  }

  @Test
  void retrieve_sparseRetriever_invokedExactlyOnce() {
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(List.of());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    retriever.retrieve(query);

    verify(sparse, times(1)).retrieve(query);
  }

  @Test
  void retrieve_joiner_invokedExactlyOnce() {
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(List.of());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    retriever.retrieve(query);

    verify(joiner, times(1)).join(any());
  }

  @Test
  void retrieve_returnsJoinerOutput() {
    // topK=10 is larger than the 3-doc result → all joiner docs come back, no extra fusion.
    List<Document> joinerResult = List.of(doc("a"), doc("b"), doc("c"));
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(joinerResult);

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder()
            .dense(dense)
            .sparse(sparse)
            .joiner(joiner)
            .topK(10)
            .build();

    assertThat(retriever.retrieve(query)).containsExactlyElementsOf(joinerResult);
  }

  // -------------------------------------------------------------------------
  // 3. Exception propagation — no fallback, no swallowing
  // -------------------------------------------------------------------------

  @Test
  void retrieve_denseThrows_propagatesUnchanged() {
    RuntimeException cause = new RuntimeException("dense down");
    when(dense.retrieve(query)).thenThrow(cause);

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    assertThatThrownBy(() -> retriever.retrieve(query)).isSameAs(cause);
    verify(sparse, never()).retrieve(any());
    verify(joiner, never()).join(any());
  }

  @Test
  void retrieve_sparseThrows_propagatesUnchanged() {
    RuntimeException cause = new RuntimeException("sparse down");
    when(dense.retrieve(query)).thenReturn(List.of(doc("d")));
    when(sparse.retrieve(query)).thenThrow(cause);

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    assertThatThrownBy(() -> retriever.retrieve(query)).isSameAs(cause);
    verify(joiner, never()).join(any());
  }

  @Test
  void retrieve_joinerThrows_propagatesUnchanged() {
    RuntimeException cause = new RuntimeException("joiner down");
    when(dense.retrieve(query)).thenReturn(List.of(doc("d")));
    when(sparse.retrieve(query)).thenReturn(List.of(doc("s")));
    when(joiner.join(any())).thenThrow(cause);

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    assertThatThrownBy(() -> retriever.retrieve(query)).isSameAs(cause);
  }

  // -------------------------------------------------------------------------
  // 4. Joiner input — Map<Query, List<List<Document>>> structure
  // -------------------------------------------------------------------------

  @Test
  @SuppressWarnings("unchecked")
  void retrieve_joinerInput_hasOneEntryWithDenseListThenSparseList() {
    List<Document> denseResults = List.of(doc("d1"), doc("d2"));
    List<Document> sparseResults = List.of(doc("s1"));
    when(dense.retrieve(query)).thenReturn(denseResults);
    when(sparse.retrieve(query)).thenReturn(sparseResults);
    when(joiner.join(any())).thenReturn(List.of());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    retriever.retrieve(query);

    ArgumentCaptor<Map<Query, List<List<Document>>>> captor = ArgumentCaptor.forClass(Map.class);
    verify(joiner).join(captor.capture());
    Map<Query, List<List<Document>>> input = captor.getValue();

    assertThat(input).hasSize(1); // exactly one Query entry
    List<List<Document>> lists = input.get(query);
    assertThat(lists).hasSize(2); // dense list + sparse list
    assertThat(lists.get(0)).isSameAs(denseResults); // dense is first
    assertThat(lists.get(1)).isSameAs(sparseResults); // sparse is second
  }

  // -------------------------------------------------------------------------
  // 5. topK behavior — HybridDocumentRetriever applies the final limit
  // -------------------------------------------------------------------------

  @Test
  void retrieve_topK_truncatesResultToConfiguredSize() {
    List<Document> fiveDocs = docs(5);
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(fiveDocs);

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder()
            .dense(dense)
            .sparse(sparse)
            .joiner(joiner)
            .topK(2)
            .build();

    List<Document> result = retriever.retrieve(query);
    assertThat(result).hasSize(2);
    assertThat(result).containsExactly(fiveDocs.get(0), fiveDocs.get(1));
  }

  @Test
  void retrieve_topK_largerThanResult_returnsAllDocuments() {
    List<Document> threeDocs = docs(3);
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(threeDocs);

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder()
            .dense(dense)
            .sparse(sparse)
            .joiner(joiner)
            .topK(10)
            .build();

    assertThat(retriever.retrieve(query)).containsExactlyElementsOf(threeDocs);
  }

  @Test
  void retrieve_defaultTopK_limitsToTen() {
    // No explicit topK: default must be exactly 10.
    // Joiner returns 15 docs → only 10 come back.
    List<Document> fifteenDocs = docs(15);
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(fifteenDocs);

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    assertThat(retriever.retrieve(query)).hasSize(10);
  }

  // -------------------------------------------------------------------------
  // 6. Query propagation — reference identity, not just equality
  // -------------------------------------------------------------------------

  @Test
  void retrieve_exactQueryInstance_passedToBothRetrievers() {
    Query specificQuery = new Query("specific text");
    when(dense.retrieve(specificQuery)).thenReturn(List.of());
    when(sparse.retrieve(specificQuery)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(List.of());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    retriever.retrieve(specificQuery);

    verify(dense).retrieve(same(specificQuery));
    verify(sparse).retrieve(same(specificQuery));
  }

  @Test
  @SuppressWarnings("unchecked")
  void retrieve_exactQueryInstance_passedToJoiner() {
    Query specificQuery = new Query("specific text");
    when(dense.retrieve(specificQuery)).thenReturn(List.of());
    when(sparse.retrieve(specificQuery)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(List.of());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    retriever.retrieve(specificQuery);

    ArgumentCaptor<Map<Query, List<List<Document>>>> captor = ArgumentCaptor.forClass(Map.class);
    verify(joiner).join(captor.capture());
    Query capturedKey = captor.getValue().keySet().iterator().next();
    assertThat(capturedKey).isSameAs(specificQuery);
  }

  // -------------------------------------------------------------------------
  // 7. Empty retrieval
  // -------------------------------------------------------------------------

  @Test
  @SuppressWarnings("unchecked")
  void retrieve_emptyResults_joinerReceivesTwoEmptyLists() {
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(List.of());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    retriever.retrieve(query);

    ArgumentCaptor<Map<Query, List<List<Document>>>> captor = ArgumentCaptor.forClass(Map.class);
    verify(joiner).join(captor.capture());
    List<List<Document>> lists = captor.getValue().get(query);
    assertThat(lists).hasSize(2);
    assertThat(lists.get(0)).isEmpty();
    assertThat(lists.get(1)).isEmpty();
  }

  @Test
  void retrieve_emptyResults_returnsEmptyList() {
    when(dense.retrieve(query)).thenReturn(List.of());
    when(sparse.retrieve(query)).thenReturn(List.of());
    when(joiner.join(any())).thenReturn(List.of());

    HybridDocumentRetriever retriever =
        HybridDocumentRetriever.builder().dense(dense).sparse(sparse).joiner(joiner).build();

    assertThat(retriever.retrieve(query)).isEmpty();
  }
}
