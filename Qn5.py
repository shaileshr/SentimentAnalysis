import re
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.clustering import KMeans


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


cleanedReviewRDD.saveAsPickleFile(path+'/P2CleanedRDD',10)

reviewRDD = sc.pickleFile('/Users/sradhakr/Desktop/Assignment3/Assignment3/P2CleanedRDD', 10)

uniqueWordsRDD = reviewRDD.flatMap(lambda words: words).distinct().map(lambda word: (word, 1))

word2VecRDD = sqlContext.read.parquet(path+"/word2vec/data")

wordsFeaturesRDD = uniqueWordsRDD.join(word2VecRDD.rdd).map(lambda (key, (dummy,features)):(key, features))

#not a RDD
kMeansclusters = KMeans.train(wordsFeaturesRDD.map(lambda (key, features): features), 2000, maxIterations=50, runs=5, initializationMode="random", seed=50)


wordsClustersRDD = wordsFeaturesRDD.map(lambda (key,features): (key,kMeansclusters.predict(features)))

wordsClustersRDD.saveAsPickleFile(path+'/WordClustersRDD',10)


clusterAndWordsListRDD = wordsClustersRDD.map(lambda (word, clusterIndex): (clusterIndex,word)).groupByKey(10).cache()

print map(lambda x: (x[0], list(x[1])), clusterAndWordsListRDD.take(10))