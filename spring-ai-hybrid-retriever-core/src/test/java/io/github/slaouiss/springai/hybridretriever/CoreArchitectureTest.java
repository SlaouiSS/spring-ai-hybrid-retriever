package io.github.slaouiss.springai.hybridretriever;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses;

import com.tngtech.archunit.junit.AnalyzeClasses;
import com.tngtech.archunit.junit.ArchTest;
import com.tngtech.archunit.lang.ArchRule;

@AnalyzeClasses(packages = "io.github.slaouiss.springai.hybridretriever")
class CoreArchitectureTest {

  @ArchTest
  static final ArchRule NO_SPRING_BOOT_IN_CORE =
      noClasses().should().dependOnClassesThat().resideInAPackage("org.springframework.boot..");

  @ArchTest
  static final ArchRule NO_LUCENE_IN_CORE =
      noClasses().should().dependOnClassesThat().resideInAPackage("org.apache.lucene..");
}
