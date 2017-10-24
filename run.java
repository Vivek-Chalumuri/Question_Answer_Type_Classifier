LexicalizedParser lp = LexicalizedParser.loadModel(
"edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",
"-maxLength", "80", "-retainTmpSubcategories");
TreebankLanguagePack tlp = new PennTreebankLanguagePack();
// Uncomment the following line to obtain original Stanford Dependencies
// tlp.setGenerateOriginalDependencies(true);
GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
String[] sent = "This", "is", "an", "easy", "sentence", "." ;
Tree parse = lp.apply(Sentence.toWordList(sent));
GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
CollectionhTypedDependencyi tdl = gs.typedDependenciesEnhancedPlusPlus();
System.out.println(tdl);
