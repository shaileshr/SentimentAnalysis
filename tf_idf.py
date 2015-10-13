import re
from pyspark.sql import SQLContext
from nltk.corpus import stopwords
sqlContext = SQLContext(sc)
import nltk
from nltk.corpus import stopwords
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import datetime

#Change the path here
path = "/Users/sradhakr/Desktop/Assignment3/Assignment3"
nltk.data.path.append(path+'/nltk_data')

stop_words = set(stopwords.words("english"))

df = sqlContext.read.json(path+'/reviews_Pet_Supplies_p1.json')

reviewDF = df.select("overall", "reviewText", "reviewTime").printSchema()

def removePunctuation(text):
	return re.sub("[^a-zA-Z]", " ", text)


cleanedReviewRDD = reviewDF.map(lambda row: (row.overall, filter(lambda word: word not in stop_words, removePunctuation(row.reviewText).lower().split()), row.reviewTime ))


hashingTF = HashingTF()

tf = hashingTF.transform(cleanedReviewRDD.map(lambda (score,review,time): review))
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

formatter_string = "%m %d %Y"

featureScoreTimeRDD = tfidf.zip(cleanedReviewRDD.map(lambda (score, rawword, time): (score,time)))\
	.map(lambda (feature, (score, time)): (feature,score, datetime.datetime.strptime(time[-4:], "%Y").year) )

train_featureScoreTimeRDD=featureScoreTimeRDD.filter(lambda (feature, score, time): time<2014 )\
	.map(lambda (feature, score, time): (feature, score)).cache()
val_featureScoreTimeRDD=featureScoreTimeRDD.filter(lambda (feature, score, time): time>=2014 )\
	.map(lambda (feature, score, time): (feature, score)).cache()

train_featureScoreTimeRDD.saveAsPickleFile(path+'trainDataRDD',10)
val_featureScoreTimeRDD.saveAsPickleFile(path+'valDataRDD',10)