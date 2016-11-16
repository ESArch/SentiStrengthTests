import unicodedata
from nltk.corpus import stopwords
from nltk import bigrams
import string
import re
import operator
from collections import Counter
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

f = open ('tweetsTuristas+results.txt', 'r', encoding='utf8')

tweets = list(f)
textos = [str(i.split('\t')[0]) for i in tweets]
positivas = [int(i.split('\t')[2]) for i in tweets]
negativas = [int(i.split('\t')[3]) for i in tweets]

tweets = zip(textos, positivas, negativas)
tweets = sorted(tweets, key=lambda x: x[2], reverse=True)


### Histogram plot

histogram = {}

for tweet in tweets:
    if abs(tweet[1]) > abs(tweet[2]):
        histogram[tweet[1]] = histogram.get(tweet[1],0) + 1
    elif abs(tweet[1]) < abs(tweet[2]):
        histogram[tweet[2]] = histogram.get(tweet[1], 0) + 1
    else:
        histogram[0] = histogram.get(0,0) + 1

print(histogram)

N = len(histogram.keys())
sentValues = sorted(histogram.keys())
counts = list()

for key in sorted(histogram.keys()):
    #print(key,histogram[key])
    counts += [histogram[key]]

ind = np.arange(N)
width = 0.4

plt.style.use('custom538')
fig,ax = plt.subplots()
fig.set_size_inches(8,8)

rect = ax.bar(ind, counts, width)

ax.set_title("Strongest sentiment")
ax.set_ylabel("Tweets")
ax.set_xticks(ind+width/2)
ax.set_xticklabels(sentValues)

plt.savefig('sentAnalisis.png')
#plt.show()


### Most positive tweets
f = open('positiveTweets.txt', 'w', encoding='utf8')

for tweet in tweets:
    if abs(tweet[1]) > abs(tweet[2]) and abs(tweet[1]) >= 3:
        f.write('\t'.join([tweet[0],str(tweet[1]),str(tweet[2])])+'\n')



### Most negative tweets
f = open('negativeTweets.txt', 'w', encoding='utf8')

for tweet in tweets:
    if abs(tweet[1]) < abs(tweet[2]) and abs(tweet[2]) >= 3:
        f.write('\t'.join([tweet[0],str(tweet[1]),str(tweet[2])])+'\n')


### Most common terms

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
    # Faltan palabras con tildes y algunos problemas con la Ñ
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def remove_diacritics(text):
    """
    Returns a string with all diacritics (aka non-spacing marks) removed.
    For example "Héllô" will become "Hello".
    Useful for comparing strings in an accent-insensitive fashion.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")

def mostFrequentTerms(tweets, nTerms ):
    # Punctuation signs
    punctuation = list(string.punctuation) + ['¿', '¡', '…']
    # stopwords + punctuation signs
    stop = stopwords.words('english') + [w.title() for w in stopwords.words('english')]\
           + [w.upper() for w in stopwords.words('english')] + [w.lower() for w in stopwords.words('english')] + punctuation

    stop += stopwords.words('spanish') + [w.title() for w in stopwords.words('spanish')] \
           + [w.upper() for w in stopwords.words('spanish')] + [w.lower() for w in
                                                                stopwords.words('spanish')] + punctuation

    # Most frequent terms
    count_all = Counter()

    for tweet in tweets:
        terms_stop = [term for term in preprocess(remove_diacritics(tweet)) if term not in stop]
        count_all.update(terms_stop)

    return count_all.most_common(nTerms)


posTweets = list()
negTweets = list()
for tweet in tweets:
    if abs(tweet[1]) > abs(tweet[2]) and abs(tweet[1]) >= 3:
        posTweets += [tweet[0]]
    elif abs(tweet[1]) < abs(tweet[2]) and abs(tweet[2]) >= 3:
        negTweets += [tweet[0]]

freqTerms = mostFrequentTerms(posTweets, 10)

N = len(freqTerms)
terms = [str(i[0]) for i in freqTerms]
counts = [int(i[1]) for i in freqTerms]

ind = np.arange(N)
width = 0.4

plt.style.use('custom538')
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 6)

rect1 = ax1.bar(ind, counts, width)

ax1.set_title("Most frequent terms in positive tweets")
ax1.set_ylabel("Occurrences")
ax1.set_xticks(ind+width/2)
ax1.set_xticklabels(terms, rotation=30, ha='right')

freqTerms = mostFrequentTerms(negTweets, 10)
terms = [str(i[0]) for i in freqTerms]
counts = [int(i[1]) for i in freqTerms]

rect2 = ax2.bar(ind, counts, width)

ax2.set_title("Most frequent terms in negative tweets")
ax2.set_ylabel("Occurrences")
ax2.set_xticks(ind+width/2)
ax2.set_xticklabels(terms, rotation=30, ha='right')

plt.savefig('commonTerms.png')
#plt.show()


'''
fw = open('tweetsAnalizados.txt', 'w', encoding='utf8')

for tweet in tweets:
    splTweet = tweet.split('\t')
    fw.write(splTweet[0] + '\t' + splTweet[2] + '\t' + splTweet[3] + '\n')
'''