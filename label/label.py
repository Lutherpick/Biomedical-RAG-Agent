
import pandas as pd
import nltk
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)  
nltk.download('omw-1.4',quiet=True)  
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from io import StringIO

import sklearn.feature_extraction.text as vec
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD

from gensim.models import LsiModel
from gensim.models import LdaModel
from gensim import corpora

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

def getChunks():
    f=open("./pmc_chunker/out/chunks.json")
    df=pd.read_json(StringIO(f.read()),lines=True,orient="records")
    return df


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

#Latent Dirichlet Allocation (LDA)
#A value close to 1 means the documents are very similar based on the words in them, whereas a value close to 0 means they're quite different.
def methode1(workingSet,numTopicWords,numTopics):

    #stop = set(stopwords.words('english'))
    #exclude = set(string.punctuation)
    #lemma = WordNetLemmatizer()


    normalized=[]
    for text in workingSet:
        stop_free = " ".join([i for i in text.lower().split() if i not in stop])
        punc_free = "".join(ch for ch in stop_free if ch not in exclude)
        #normalized.append((" ".join(lemma.lemmatize(word) for word in punc_free.split()).split()))
        normalized.append(" ".join(lemma.lemmatize(word) for word in punc_free.split()))
        #print(normalized[0])





    tfModel = vec.TfidfVectorizer(stop_words=[],use_idf=False)
    tfTrained = tfModel.fit(normalized)
    X = tfTrained.transform(normalized)

    ldamodel=LatentDirichletAllocation(n_components=numTopics,learning_method='online',random_state=42,max_iter=6)

    ldaTrained=ldamodel.fit(X)
    #lda_top=ldaTrained.transform(X)



    
    topics=[]
    vocab=tfModel.get_feature_names_out()
    for i in ldaTrained.components_:
        temp=zip(vocab,i)

        labels=sorted(temp, key= lambda x:x[1], reverse=True)[:numTopicWords]
        # strin=[]
        # num=0
        # for element in labels:
        #     strin.append(element[0])
        #     num=num+element[1]
        # print(str(strin)+" "+str(num))
        # print(labels)
        topics.append(labels)

    #print("methode1")
    return topics




def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


#Latent Semantic Analysis (LSA)
def methode2(workingSet,numTopicWords,numTopics):

    clean_corpus = [clean(doc).split() for doc in workingSet]
    dictionary = corpora.Dictionary(clean_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]
    #print(doc_term_matrix)
    lsa = LsiModel(doc_term_matrix, num_topics=numTopics, id2word = dictionary)

    #print(lsa.print_topics(num_topics=numTopics, num_words=numTopicWords))
    #print(lsa.show_topics(num_topics=numTopics, num_words=numTopicWords))

    topics=[]
    for i in range(numTopics):
        if(i>numTopics):
            break
        topics.append(lsa.show_topic(i)[:numTopicWords])
     


    #print("methode2")
    #return lsa[doc_term_matrix[0]]#lsa.id2word[100]
    return topics

#LSA
def methode3(workingSet,numTopicWords,numTopics):

    #stop = set(stopwords.words('english'))
    #exclude = set(string.punctuation)
    #lemma = WordNetLemmatizer()


    normalized=[]
    for text in workingSet:
        stop_free = " ".join([i for i in text.lower().split() if i not in stop])
        punc_free = "".join(ch for ch in stop_free if ch not in exclude)
        #normalized.append((" ".join(lemma.lemmatize(word) for word in punc_free.split()).split()))
        normalized.append(" ".join(lemma.lemmatize(word) for word in punc_free.split()))
        #print(normalized[0])


    tfModel = vec.TfidfVectorizer(stop_words=[],use_idf=True)
    tfTrained = tfModel.fit(normalized)
    X = tfTrained.transform(normalized)

    ldamodel=TruncatedSVD(n_components=numTopics,algorithm='randomized',random_state=42,n_iter=6)
    #ldamodel=TruncatedSVD(n_components=numTopics,algorithm='arpack',random_state=42,n_iter=6)

    ldaTrained=ldamodel.fit(X)
    #lda_top=ldaTrained.transform(X)



    
    topics=[]
    vocab=tfModel.get_feature_names_out()
    for i in ldaTrained.components_:
        temp=zip(vocab,i)

        labels=sorted(temp, key= lambda x:x[1], reverse=True)[:numTopicWords]

        topics.append(labels)

    #print("methode3")
    return topics


#Latent Dirichlet Allocation (LDA)
def methode4(workingSet,numTopicWords,numTopics):

    clean_corpus = [clean(doc).split() for doc in workingSet]
    dictionary = corpora.Dictionary(clean_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]
    #print(doc_term_matrix)
    lsa = LdaModel(doc_term_matrix, num_topics=numTopics, id2word = dictionary)

    #print(lsa.print_topics(num_topics=numTopics, num_words=numTopicWords))
    #print(lsa.show_topics(num_topics=numTopics, num_words=numTopicWords))

    topics=[]
    for i in range(numTopics):
        if(i>numTopics):
            break
        topics.append(lsa.show_topic(i)[:numTopicWords])
     


    #print("methode2")
    #return lsa[doc_term_matrix[0]]#lsa.id2word[100]
    return topics

def methode5(workingSet,numTopicWords,numTopics):


    normalized=[]
    for text in workingSet:
        stop_free = " ".join([i for i in text.lower().split() if i not in stop])
        punc_free = "".join(ch for ch in stop_free if ch not in exclude)
        #normalized.append((" ".join(lemma.lemmatize(word) for word in punc_free.split()).split()))
        normalized.append(" ".join(lemma.lemmatize(word) for word in punc_free.split()))
        #print(normalized[0])

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(normalized)
    
    retTopics=[]
    for i in range(numTopics):
        if(i>numTopics):
            break
        retTopics.append(topic_model.get_topic(0)[:numTopicWords])
    return retTopics

#bert supervised labeling
def methode6(workingSet,numTopicWords,numTopics):
    #tutorial: https://maartengr.github.io/BERTopic/getting_started/manual/manual.html
    print("not implemented yet")

    # normalized=[]
    # for text in workingSet:
        # stop_free = " ".join([i for i in text.lower().split() if i not in stop])
        # punc_free = "".join(ch for ch in stop_free if ch not in exclude)
        # #normalized.append((" ".join(lemma.lemmatize(word) for word in punc_free.split()).split()))
        # normalized.append(" ".join(lemma.lemmatize(word) for word in punc_free.split()))
        # #print(normalized[0])

    # topic_model = BERTopic()
    # topics, probs = topic_model.fit_transform(normalized)
    
    # retTopics=[]
    # for i in range(numTopics):
        # if(i>numTopics):
            # break
        # retTopics.append(topic_model.get_topic(0)[:numTopicWords])
    # return retTopics



if __name__ == '__main__':

    df=getChunks()
    workingset=df[-200:-100]["chunk_text"]
    # #workingset=['study identified distinct behavioral pattern transition path unfamiliar pig agonistic encounter experiment 1 highlighted short exploratory pathway looking → attacking scenario unfamiliar pig engaged direct confrontation contrast experiment 2 involved introducing new pig established group revealed complex prolonged exploratory pathway often involving looking sniffing touching attacking analysis behavioral transition path revealed eight distinct type path culminating ritualized nonaggressive interaction others led aggressive outcome finding suggest pig often use nonaggressive behavior establish social order aggressive interaction likely ritualized behavior insufficient resolving conflict overall study underscore importance understanding intricacy behavioral transition pig play crucial role shaping social dynamic within group identification specific exploratory pathway provides insight pig navigate social interaction balancing exploration aggression based context encounter', 'figure 1 method classifying dominance hierarchy information enhance understanding animal social structure behavioral pattern various path leading aggressive behavior systematically summarized first time furthermore dominance hierarchy information within group meticulously classified', 'figure 2 ethogram resident pig encountering newly added pig resident pig locked eye onto added pig b resident pig sniffed odor body added pig c resident pig touched prodded added pig’s body nose one resident pig opened mouth bite added pig high resolution see supplementary material', 'figure 3 ethogram newly added pig responding resident pig newly added pig fought back b newly added pig hid within resident group c newly added pig urinated', 'figure 4 behavioral transition path unfamiliar pig agonistic meeting short exploratory path observed experiment 1 b longest exploratory path observed experiment 2 c several type shortlong exploratory path experiment 2']
    #workingset=['study identified distinct behavioral pattern transition path unfamiliar pig agonistic encounter experiment 1 highlighted short exploratory pathway looking → attacking scenario unfamiliar pig engaged direct confrontation contrast experiment 2 involved introducing new pig established group revealed complex prolonged exploratory pathway often involving looking sniffing touching attacking analysis behavioral transition path revealed eight distinct type path culminating ritualized nonaggressive interaction others led aggressive outcome finding suggest pig often use nonaggressive behavior establish social order aggressive interaction likely ritualized behavior insufficient resolving conflict overall study underscore importance understanding intricacy behavioral transition pig play crucial role shaping social dynamic within group identification specific exploratory pathway provides insight pig navigate social interaction balancing exploration aggression based context encounter', 'figure 1 method classifying dominance hierarchy information enhance understanding animal social structure behavioral pattern various path leading aggressive behavior systematically summarized first time furthermore dominance hierarchy information within group meticulously classified', 'figure 2 ethogram resident pig encountering newly added pig resident pig locked eye onto added pig b resident pig sniffed odor body added pig c resident pig touched prodded added pig’s body nose one resident pig opened mouth bite added pig high resolution see supplementary material', 'figure 3 ethogram newly added pig responding resident pig newly added pig fought back b newly added pig hid within resident group c newly added pig urinated', 'figure 4 behavioral transition path unfamiliar pig agonistic meeting short exploratory path observed experiment 1 b longest exploratory path observed experiment 2 c several type shortlong exploratory path experiment 2']
    temp=methode1(workingset,3,3)
    print()#"sklearn LDA")
    print(temp)
    
    temp2=methode2(workingset,3,3)
    print()#"ginsim LDA")
    print(temp2)

    temp3=methode3(workingset,3,3)
    print()#"sklearn LSA")
    print(temp3)

    temp4=methode4(workingset,3,3)
    print()#"ginsim LSA")
    print(temp4)

    temp5=methode5(workingset,3,2)
    print()
    print(temp5)