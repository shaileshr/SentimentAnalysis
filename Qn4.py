import re
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel 


conf = SparkConf()

conf.setMaster('yarn-client')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

path = "/Users/sradhakr/Desktop/Assignment3/Assignment3"

df = sqlContext.read.json(path+'/reviews_Pet_Supplies_p2.json')

reviewDF = df.select("reviewText")

sc = SparkContext(appName='Word2Vec')

def removePunctuation(text):
   return re.sub("[^a-zA-Z]", " ", text)


cleanedReviewRDD = reviewDF.map(lambda row: removePunctuation(row.reviewText).lower().split())

word2vec = Word2Vec()
model = word2vec.fit(cleanedReviewRDD)

model.save(sc , path+"/word2vec")

synonyms,  = model.findSynonyms('pet', 40)
print synonyms

sameModel = sqlContext.read.parquet(path+"/word2vec/data")