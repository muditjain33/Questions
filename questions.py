import nltk
import sys
import os
import string
import math

from nltk.util import pr

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    d=dict()
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        f=open(file_path, encoding="utf8")
        data = f.read()
        d[file_name]=data
    return d


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    l=[]
    listen=nltk.tokenize.word_tokenize(document)
    stwords=nltk.corpus.stopwords.words("english")
    punc=list(string.punctuation)
    for i in listen:
        i=i.lower()
        if i in stwords:
            continue
        if i in punc:
            continue
        if i=='':
            continue
        d=''
        for j in i:
            if j not in punc:
                d=d+j
        if d=='':
            continue
        l.append(d)
    return l


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    new_d=dict()
    new_s=set()
    nod=0
    for k,v in documents.items():
        new_s.update(v)
        nod=nod+1
    for i in new_s:
        no=0
        for k,v in documents.items():
            if(i in v):
                no=no+1
        new_d[i]=math.log(nod/no)
    return new_d


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    new_f=dict()
    for k,v in files.items():
        new_f[k]=0
    for i in query:
        if i not in idfs.keys():
            continue
        for k,v in files.items():
            new_f[k]+=(idfs[i]*(v.count(i)))

    sort_value = dict(sorted(new_f.items(), key=lambda item: item[1],reverse=True))
    l=[]
    for k,v in sort_value.items():
        l.append(k)
        n=n-1
        if(n==0):
            break
    return l

def compare(e):
  return (e[1],e[2])

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    new_s=dict()
    for k,v in sentences.items():
        new_s[k]=0
    for i in query:
        if i not in idfs.keys():
            continue
        for k,v in sentences.items():
            if(i in v):
                new_s[k]+=idfs[i]

    sort_value = dict(sorted(new_s.items(), key=lambda item: item[1],reverse=True))
    l=[]
    for k,v in sort_value.items():
        w=[]
        w.append(k)
        w.append(v)
        
        wordskilist=sentences[k]
        no=0
        to=0
        for i in wordskilist:
            to+=1
            if(i in query):
                no+=1
        to=no/to
        w.append(to)
        l.append(w)
    l= sorted(l,key=compare,reverse=True)
    nl=[]
    for i in l:
        nl.append(i[0])
        n=n-1
        if(n==0):
            break        
    return nl



if __name__ == "__main__":
    main()