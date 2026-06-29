package io.github.slaouiss.springai.hybridretriever.demo;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

/** BM25 sparse retriever backed by a Lucene in-memory index. */
public class LuceneDocumentRetriever implements DocumentRetriever {

  private final IndexSearcher searcher;
  private final StandardAnalyzer analyzer;
  private final Map<String, Document> documentsById;
  private final int topK;

  public LuceneDocumentRetriever(List<Document> documents, int topK) throws Exception {
    this.topK = topK;
    this.analyzer = new StandardAnalyzer();
    this.documentsById =
        documents.stream().collect(Collectors.toMap(Document::getId, Function.identity()));

    var directory = new ByteBuffersDirectory();
    var config = new IndexWriterConfig(analyzer);
    try (var writer = new IndexWriter(directory, config)) {
      for (var doc : documents) {
        var luceneDoc = new org.apache.lucene.document.Document();
        luceneDoc.add(new TextField("content", doc.getText(), Field.Store.NO));
        luceneDoc.add(new StoredField("id", doc.getId()));
        writer.addDocument(luceneDoc);
      }
    }
    this.searcher = new IndexSearcher(DirectoryReader.open(directory));
  }

  @Override
  public List<Document> retrieve(Query query) {
    try {
      var parser = new QueryParser("content", analyzer);
      var luceneQuery = parser.parse(QueryParser.escape(query.text()));
      var hits = searcher.search(luceneQuery, topK);
      var storedFields = searcher.storedFields();
      var results = new ArrayList<Document>();
      for (var scoreDoc : hits.scoreDocs) {
        var id = storedFields.document(scoreDoc.doc).get("id");
        var doc = documentsById.get(id);
        if (doc != null) results.add(doc.mutate().score((double) scoreDoc.score).build());
      }
      return results;
    } catch (Exception e) {
      throw new RuntimeException("Lucene retrieval failed for: " + query.text(), e);
    }
  }
}
