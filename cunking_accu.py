import nltk
from nltk.corpus import state_union, conll2000
from nltk.tokenize import PunktSentenceTokenizer

# Training data
train_text = state_union.raw("2005-GWBush.txt")
train_tokenized = PunktSentenceTokenizer(train_text)

# Testing data
test_text = conll2000.raw("test.txt")
test_tokenized = test_text.split("\n")

# Function to perform chunking
def chunking():
    # Training the chunker
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(train_text)

    # Defining the chunk grammar
    chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

    # Creating the chunk parser
    chunkParser = nltk.RegexpParser(chunkGram)

    # Chunking each sentence
    chunked = []
    for sent in tokenized:
        tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        tree = chunkParser.parse(tagged)
        chunked.append(tree)

    # Evaluating the chunker
    iob_tagged = []
    for sent in test_tokenized:
        tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        tree = chunkParser.parse(tagged)
        iob_tagged.append(nltk.chunk.tree2conlltags(tree))

    # Calculating the performance
    reference = []
    test = []
    for i in range(len(iob_tagged)):
        for j in range(len(iob_tagged[i])):
            reference.append(iob_tagged[i][j][2])
            test.append(iob_tagged[i][j][1])
    accuracy = nltk.metrics.accuracy(reference, test)
    precision = nltk.metrics.precision(set(reference), set(test))
    recall = nltk.metrics.recall(set(reference), set(test))
    f_measure = nltk.metrics.f_measure(set(reference), set(test))

    # Printing the results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-Measure:", f_measure)

# Function to perform chinking
def chinking():
    # Training the chunker
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(train_text)

    # Defining the chink grammar
    chinkGram = r"""Chunk: {<.*>+}
                    }<VB.?|IN|DT|TO>+{"""

    # Creating the chink parser
    chinkParser = nltk.RegexpParser(chinkGram)

    # Chunking each sentence
    chunked = []
    for sent in tokenized:
        tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        tree = chinkParser.parse(tagged)
        chunked.append(tree)

    # Evaluating the chunker
    iob_tagged = []
    for sent in test_tokenized:
        tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        tree = chinkParser.parse(tagged)
        iob_tagged.append(nltk.chunk.tree2conlltags(tree))

    # Calculating the performance
    reference = []
    test = []
    for i in range(len(iob_tagged)):
        for j in range(len(iob_tagged[i])):
            reference.append(iob_tagged[i][j][2])
            test.append(iob_tagged[i][j][1])
    accuracy = nltk.metrics.accuracy(reference, test)
    precision = nltk.metrics.precision(set(reference))
chunking()
chinking()