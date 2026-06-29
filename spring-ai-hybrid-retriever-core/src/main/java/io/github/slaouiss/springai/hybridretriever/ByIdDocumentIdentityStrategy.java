package io.github.slaouiss.springai.hybridretriever;

import org.springframework.ai.document.Document;

/**
 * {@link DocumentIdentityStrategy} that uses {@link
 * org.springframework.ai.document.Document#getId()} as the identity key.
 *
 * <p>This is the default strategy used by {@link RRFDocumentJoiner}: two documents from different
 * sources are fused only when they share the same id.
 */
public final class ByIdDocumentIdentityStrategy implements DocumentIdentityStrategy {

  @Override
  public Object identity(Document document) {
    return document.getId();
  }
}
