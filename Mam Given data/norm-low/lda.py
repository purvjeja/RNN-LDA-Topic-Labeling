from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import glob

#Tokenizing: converting a document to its atomic elements
#Stopping: removing meaningless words
#Stemming: merging words that are equivalent in meaning

tokenizer = RegexpTokenizer(r'\w+')#Regular-Expression Tokenizers -> splits a string into substrings using a regular expression

# create English stop words list
en_stop = get_stop_words('english')#meaningless words in english eg: is, are, the

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()#removing similar objects

doc_set = list()
#sample documents
files = glob.glob('*.txt')
for file_name in files:
    readall = open(file_name,'r')
    doc_set.append(readall.read())
#print(doc_set)
#doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
#doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
#doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
#doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
#doc_e = "Health professionals say that brocolli is good for your health." 

# compile sample documents into a list
#doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:# clean and tokenize document string
    raw = i.lower()
    #print ('raw\n%s'%raw)
    tokens = tokenizer.tokenize(raw)# remove stop words from tokens
    #print ('tokens\n%s'%tokens)
    stopped_tokens = [i for i in tokens if not i in en_stop]# stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]# add tokens to list
    texts.append(stopped_tokens)
print ('tokens = %s \n'%tokens)
print ('stopped_tokens = %s \n'%stopped_tokens)
print ('stemmed_tokens = %s \n'%stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)
print (len(dictionary))
for j in  dictionary:
    print (dictionary[j])
#print (dictionary)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
print ('\n')
print (corpus)
print ('\n')
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=1, num_words=len(dictionary)))


