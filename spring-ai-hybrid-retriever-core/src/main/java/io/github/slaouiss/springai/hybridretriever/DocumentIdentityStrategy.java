package io.github.slaouiss.springai.hybridretriever;

import org.springframework.ai.document.Document;

/**
 * Strategy that assigns an identity key to a {@link org.springframework.ai.document.Document}.
 *
 * <p>Two documents with equal keys are treated as the same logical document: their scores are
 * accumulated across all ranked lists in which they appear. The default implementation uses {@link
 * org.springframework.ai.document.Document#getId()}.
 */
@FunctionalInterface
public interface DocumentIdentityStrategy {

  /**
   * Returns the identity key for {@code document}.
   *
   * @param document the document to identify; never {@code null}
   * @return a non-null identity key; two documents with equal keys are treated as one
   */
  Object identity(Document document);
}
