import spacy
import stanza

# Load the Spacy and Spacy-Stanza parsers
nlp_spacy = spacy.load('en_core_web_sm')
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

# Define the sentence
sentence = 'The doctor gave the spotted lemon to the doctor'

# Parse the sentence using Spacy and Spacy-Stanza parsers
doc_spacy = nlp_spacy(sentence)
doc_stanza = nlp_stanza(sentence)

# Print the dependency relations for each parser
for token in doc_spacy:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])

for sentence in doc_stanza.sentences:
    for word in sentence.words:
        print(word.text, word.deprel, word.head)
