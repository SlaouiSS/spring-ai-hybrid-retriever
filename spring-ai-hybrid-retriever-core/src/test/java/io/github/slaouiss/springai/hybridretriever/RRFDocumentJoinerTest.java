package io.github.slaouiss.springai.hybridretriever;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

class RRFDocumentJoinerTest {

  private static final ByIdDocumentIdentityStrategy BY_ID = new ByIdDocumentIdentityStrategy();
  private static final int DEFAULT_K = 60;
  private static final Query QUERY = new Query("q");

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private static Document doc(String id) {
    return Document.builder().id(id).text(id).build();
  }

  @SafeVarargs
  private static List<Document> join(RRFDocumentJoiner joiner, List<Document>... lists) {
    return joiner.join(Map.of(QUERY, List.of(lists)));
  }

  private static double rrfScore(int k, int rank) {
    return 1.0 / (k + rank);
  }

  private static Map<String, Double> scoreById(List<Document> docs) {
    return docs.stream().collect(Collectors.toMap(Document::getId, Document::getScore));
  }

  // -------------------------------------------------------------------------
  // 1. Disjoint ranked lists
  // -------------------------------------------------------------------------

  @Test
  void disjointLists_allDocumentsAreReturned() {
    Document a = doc("a"), b = doc("b"), c = doc("c"), d = doc("d");

    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(a, b), List.of(c, d));

    assertThat(result).hasSize(4);
    assertThat(result).extracting(Document::getId).containsExactlyInAnyOrder("a", "b", "c", "d");
  }

  @Test
  void disjointLists_eachDocumentScoresExactlyOneOverKPlusRank() {
    // a: rank 1 in list1 only  → 1/(60+1)
    // b: rank 2 in list1 only  → 1/(60+2)
    // c: rank 1 in list2 only  → 1/(60+1)
    Document a = doc("a"), b = doc("b"), c = doc("c");

    Map<String, Double> scores =
        scoreById(join(new RRFDocumentJoiner(BY_ID), List.of(a, b), List.of(c)));

    assertThat(scores.get("a")).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
    assertThat(scores.get("b")).isCloseTo(rrfScore(DEFAULT_K, 2), within(1e-10));
    assertThat(scores.get("c")).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
  }

  @Test
  void disjointLists_rankingFollowsInputOrder() {
    // Single source: a→rank1 (1/61), b→rank2 (1/62), c→rank3 (1/63). Clear descending order.
    Document a = doc("a"), b = doc("b"), c = doc("c");

    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(a, b, c), List.of());

    assertThat(result).extracting(Document::getId).containsExactly("a", "b", "c");
    assertThat(result.get(0).getScore()).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
    assertThat(result.get(1).getScore()).isCloseTo(rrfScore(DEFAULT_K, 2), within(1e-10));
    assertThat(result.get(2).getScore()).isCloseTo(rrfScore(DEFAULT_K, 3), within(1e-10));
  }

  // -------------------------------------------------------------------------
  // 2. Same document in both lists — contributions summed
  // -------------------------------------------------------------------------

  @Test
  void sharedDocument_appearsOnlyOnce() {
    Document shared = doc("shared"), other = doc("other");

    List<Document> result =
        join(new RRFDocumentJoiner(BY_ID), List.of(shared), List.of(shared, other));

    assertThat(result).hasSize(2);
    assertThat(result).extracting(Document::getId).containsExactlyInAnyOrder("shared", "other");
  }

  @Test
  void sharedDocument_scoreIsExactSumOfBothContributions() {
    Document shared = doc("shared"), other = doc("other");

    // shared: rank 1 in list1 + rank 1 in list2  → 1/61 + 1/61 = 2/61
    // other:  rank 2 in list2 only               → 1/62
    Map<String, Double> scores =
        scoreById(join(new RRFDocumentJoiner(BY_ID), List.of(shared), List.of(shared, other)));

    assertThat(scores.get("shared"))
        .isCloseTo(rrfScore(DEFAULT_K, 1) + rrfScore(DEFAULT_K, 1), within(1e-10));
    assertThat(scores.get("other")).isCloseTo(rrfScore(DEFAULT_K, 2), within(1e-10));
  }

  @Test
  void monoSourceDocument_scoreIsSingleTerm() {
    // A document present in only one list gets exactly one RRF term, no sum.
    Document solo = doc("solo");

    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(solo), List.of());

    assertThat(result).hasSize(1);
    assertThat(result.get(0).getScore()).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
  }

  // -------------------------------------------------------------------------
  // 3. DocumentIdentityStrategy
  // -------------------------------------------------------------------------

  @Test
  void customIdentityStrategy_mergesDocumentsWithSameKey() {
    Document fromDense =
        Document.builder().id("dense-1").text("t").metadata("sourceId", "article-42").build();
    Document fromSparse =
        Document.builder().id("sparse-1").text("t").metadata("sourceId", "article-42").build();

    DocumentIdentityStrategy bySourceId = d -> d.getMetadata().get("sourceId");

    List<Document> result =
        join(new RRFDocumentJoiner(bySourceId), List.of(fromDense), List.of(fromSparse));

    assertThat(result).hasSize(1);
    assertThat(result.get(0).getScore())
        .isCloseTo(rrfScore(DEFAULT_K, 1) + rrfScore(DEFAULT_K, 1), within(1e-10));
  }

  @Test
  void defaultIdentityStrategy_keepsDocumentsWithDifferentIdsSeparate() {
    Document fromDense =
        Document.builder().id("dense-1").text("t").metadata("sourceId", "article-42").build();
    Document fromSparse =
        Document.builder().id("sparse-1").text("t").metadata("sourceId", "article-42").build();

    List<Document> result =
        join(new RRFDocumentJoiner(BY_ID), List.of(fromDense), List.of(fromSparse));

    assertThat(result).hasSize(2);
  }

  // -------------------------------------------------------------------------
  // 4. RRF constant k
  // -------------------------------------------------------------------------

  @Test
  void customK_scoresEqualOneOverKPlusRank() {
    Document doc = doc("d");
    int customK = 10;

    double score =
        join(new RRFDocumentJoiner(customK, BY_ID), List.of(doc), List.of()).get(0).getScore();

    assertThat(score).isCloseTo(rrfScore(customK, 1), within(1e-10));
  }

  @Test
  void higherK_producesLowerScore() {
    Document doc = doc("d");

    double scoreK60 =
        join(new RRFDocumentJoiner(60, BY_ID), List.of(doc), List.of()).get(0).getScore();
    double scoreK120 =
        join(new RRFDocumentJoiner(120, BY_ID), List.of(doc), List.of()).get(0).getScore();

    assertThat(scoreK60).isGreaterThan(scoreK120);
    assertThat(scoreK60).isCloseTo(1.0 / 61, within(1e-10));
    assertThat(scoreK120).isCloseTo(1.0 / 121, within(1e-10));
  }

  // -------------------------------------------------------------------------
  // 5. Output sorted by descending RRF score
  // -------------------------------------------------------------------------

  @Test
  void resultsAreSortedByDescendingScore() {
    Document a = doc("a"), b = doc("b"), c = doc("c");

    // list1=[a, b, c], list2=[b]
    // b: rank 2 in list1 + rank 1 in list2  → 1/62 + 1/61  ≈ 0.0325  (highest)
    // a: rank 1 in list1 only               → 1/61          ≈ 0.0164
    // c: rank 3 in list1 only               → 1/63          ≈ 0.0159  (lowest)
    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(a, b, c), List.of(b));

    assertThat(result).extracting(Document::getId).containsExactly("b", "a", "c");
    assertThat(result.get(0).getScore())
        .isCloseTo(rrfScore(DEFAULT_K, 2) + rrfScore(DEFAULT_K, 1), within(1e-10));
    assertThat(result.get(1).getScore()).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
    assertThat(result.get(2).getScore()).isCloseTo(rrfScore(DEFAULT_K, 3), within(1e-10));
  }

  @Test
  void ties_produceSameScore() {
    // a and b each appear at rank 1 in their respective list only — tied score.
    Document a = doc("a"), b = doc("b");

    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(a), List.of(b));

    assertThat(result).hasSize(2);
    assertThat(result.get(0).getScore()).isCloseTo(result.get(1).getScore(), within(1e-10));
    assertThat(result.get(0).getScore()).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
  }

  // -------------------------------------------------------------------------
  // 6. Within-source duplicates — best rank only, never summed
  // -------------------------------------------------------------------------

  @Test
  void withinSourceDuplicate_onlyBestRankContributes() {
    // dense=[a(rank1), a(rank2), b(rank3)], sparse=[]
    // a appears twice in the SAME list — contributions must NOT be summed.
    // Only the best rank (rank 1) counts: score(a) = 1/(60+1).
    // Wrong implementation would produce 1/(60+1) + 1/(60+2) ≈ 0.0326 instead of 1/61 ≈ 0.0164.
    Document a1 = doc("a"); // rank 1
    Document a2 = doc("a"); // rank 2 — same identity, must NOT add a second term
    Document b = doc("b"); // rank 3

    Map<String, Double> scores =
        scoreById(join(new RRFDocumentJoiner(BY_ID), List.of(a1, a2, b), List.of()));

    assertThat(scores).hasSize(2);
    assertThat(scores.get("a")).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
    assertThat(scores.get("b")).isCloseTo(rrfScore(DEFAULT_K, 3), within(1e-10));
  }

  @Test
  void withinSourceDuplicate_scoreIsNotTheSumOfBothRanks() {
    Document a1 = doc("a");
    Document a2 = doc("a");

    double score = join(new RRFDocumentJoiner(BY_ID), List.of(a1, a2), List.of()).get(0).getScore();

    double wrongSummedScore = rrfScore(DEFAULT_K, 1) + rrfScore(DEFAULT_K, 2);
    assertThat(score).isNotCloseTo(wrongSummedScore, within(1e-10));
    assertThat(score).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
  }

  // -------------------------------------------------------------------------
  // 8. Dense list empty
  // -------------------------------------------------------------------------

  @Test
  void denseListEmpty_allSparseDocumentsAreReturned() {
    Document a = doc("a"), b = doc("b"), c = doc("c");

    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(), List.of(a, b, c));

    assertThat(result).hasSize(3);
    assertThat(result).extracting(Document::getId).containsExactlyInAnyOrder("a", "b", "c");
  }

  @Test
  void denseListEmpty_scoresAndOrderAreCorrect() {
    Document a = doc("a"), b = doc("b"), c = doc("c");

    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(), List.of(a, b, c));

    assertThat(result).extracting(Document::getId).containsExactly("a", "b", "c");
    assertThat(result.get(0).getScore()).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
    assertThat(result.get(1).getScore()).isCloseTo(rrfScore(DEFAULT_K, 2), within(1e-10));
    assertThat(result.get(2).getScore()).isCloseTo(rrfScore(DEFAULT_K, 3), within(1e-10));
  }

  // -------------------------------------------------------------------------
  // 9. Sparse list empty
  // -------------------------------------------------------------------------

  @Test
  void sparseListEmpty_allDenseDocumentsAreReturned() {
    Document a = doc("a"), b = doc("b"), c = doc("c");

    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(a, b, c), List.of());

    assertThat(result).hasSize(3);
    assertThat(result).extracting(Document::getId).containsExactlyInAnyOrder("a", "b", "c");
  }

  @Test
  void sparseListEmpty_scoresAndOrderAreCorrect() {
    Document a = doc("a"), b = doc("b"), c = doc("c");

    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(a, b, c), List.of());

    assertThat(result).extracting(Document::getId).containsExactly("a", "b", "c");
    assertThat(result.get(0).getScore()).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
    assertThat(result.get(1).getScore()).isCloseTo(rrfScore(DEFAULT_K, 2), within(1e-10));
    assertThat(result.get(2).getScore()).isCloseTo(rrfScore(DEFAULT_K, 3), within(1e-10));
  }

  // -------------------------------------------------------------------------
  // 10. Both lists empty
  // -------------------------------------------------------------------------

  @Test
  void bothListsEmpty_returnsEmptyListWithoutException() {
    List<Document> result = join(new RRFDocumentJoiner(BY_ID), List.of(), List.of());

    assertThat(result).isEmpty();
  }

  // -------------------------------------------------------------------------
  // 11. Default constructors
  // -------------------------------------------------------------------------

  @Test
  void defaultConstructor_usesK60AndByIdStrategy() {
    Document a = doc("a");

    List<Document> result = join(new RRFDocumentJoiner(), List.of(a), List.of());

    assertThat(result).hasSize(1);
    assertThat(result.get(0).getScore()).isCloseTo(rrfScore(DEFAULT_K, 1), within(1e-10));
  }

  @Test
  void intKConstructor_usesCustomKAndByIdStrategy() {
    Document a = doc("a"), b = doc("b");
    int customK = 10;

    // Both documents have different ids — ByIdDocumentIdentityStrategy keeps them separate.
    List<Document> result = join(new RRFDocumentJoiner(customK), List.of(a), List.of(b));

    assertThat(result).hasSize(2);
    assertThat(result.get(0).getScore()).isCloseTo(rrfScore(customK, 1), within(1e-10));
  }

  @Test
  void defaultConstructor_mergesByIdLikeByIdStrategy() {
    Document shared = doc("shared"), other = doc("other");

    // No-arg constructor must use ByIdDocumentIdentityStrategy: same id → merged.
    List<Document> result = join(new RRFDocumentJoiner(), List.of(shared), List.of(shared, other));

    assertThat(result).hasSize(2);
    Map<String, Double> scores = scoreById(result);
    assertThat(scores.get("shared"))
        .isCloseTo(rrfScore(DEFAULT_K, 1) + rrfScore(DEFAULT_K, 1), within(1e-10));
  }

  // -------------------------------------------------------------------------
  // 12. Constructor argument validation
  // -------------------------------------------------------------------------

  @Test
  void constructor_nullStrategy_throwsNullPointerException() {
    assertThatThrownBy(() -> new RRFDocumentJoiner((DocumentIdentityStrategy) null))
        .isInstanceOf(NullPointerException.class);
  }

  @Test
  void constructor_zeroK_throwsIllegalArgumentException() {
    assertThatThrownBy(() -> new RRFDocumentJoiner(0, BY_ID))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("k");
  }

  @Test
  void constructor_negativeK_throwsIllegalArgumentException() {
    assertThatThrownBy(() -> new RRFDocumentJoiner(-1, BY_ID))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("k");
  }

  // -------------------------------------------------------------------------
  // 12. Null identity key returned by strategy
  // -------------------------------------------------------------------------

  @Test
  void join_strategyReturnsNull_throwsIllegalArgumentException() {
    DocumentIdentityStrategy nullReturning = d -> null;

    assertThatThrownBy(
            () -> join(new RRFDocumentJoiner(nullReturning), List.of(doc("x")), List.of()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("null");
  }
}
