import re
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel 
import datetime
import numpy as np
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

conf = SparkConf()

conf.setMaster('yarn-client')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

path = "/Users/sradhakr/Desktop/Assignment3/Assignment3"


df = sqlContext.read.json(path+'/reviews_Pet_Supplies_p1.json')

reviewDF = df.select("overall", "reviewText", "reviewTime")

def removePunctuation(text):
   return re.sub("[^a-zA-Z]", " ", text)


cleanedReviewRDD = reviewDF.map(lambda row: (row.overall, removePunctuation(row.reviewText).lower().split(), row.reviewTime ))

reviewRDD = sc.pickleFile(path+'/P2CleanedRDD', 10)

uniqueWordsRDD = reviewRDD.flatMap(lambda words: words).distinct().map(lambda word: (word, 1))

word2VecRDD = sqlContext.read.parquet(path+"/word2vec/data")

wordsFeaturesDict = sc.broadcast(uniqueWordsRDD.join(word2VecRDD.rdd).map(lambda (key, (dummy,features)):(key, features)).collectAsMap())


def getFeature(word):
	if wordsFeaturesDict.value.has_key(word):
		return np.array(wordsFeaturesDict.value[word])
	else:
		return []	
	

reviewRDDWithIndex = cleanedReviewRDD.zipWithIndex().map(lambda (row, index):(index, row)).cache()
reviewTextWithIndex = reviewRDDWithIndex.map(lambda (index,(score, words, time)): (index, words))
reviewScoreTimeWithIndex = reviewRDDWithIndex.map(lambda (index,(score, words, time)): (index, (score, time)))


reviewFeaturesRDD = reviewTextWithIndex\
	.flatMapValues(lambda word: word)\
	.map(lambda (rowNum, word): (rowNum, (getFeature(word),1)))\
	.filter(lambda (row, (feature, dummy)): len(feature)>0)\
	.reduceByKey(lambda (x,dummy1), (y,dummy2): (x+y,dummy1+dummy2))\
	.map(lambda (rowNum, (vector,sum)): (rowNum, np.divide(vector, float(sum))))\
	.cache()
	
formatter_string = "%m %d %Y" 	
allDataRDD = reviewFeaturesRDD.join(reviewScoreTimeWithIndex)\
    .map(lambda (index, (features, (score,time))): (score, features, datetime.datetime.strptime(time[-4:], "%Y").year)).cache()

train_featureScoreTimeRDD=allDataRDD.filter(lambda (score, feature, time): time<2014 )\
    .map(lambda (score, feature, time): LabeledPoint(float(score),feature)).repartition(10).cache()
val_featureScoreTimeRDD=allDataRDD.filter(lambda (score, feature, time): time>=2014 )\
    .map(lambda (score, feature, time): LabeledPoint(float(score),feature)).repartition(10).cache()
allDataRDD.unpersist()

def getRMSE(step_array):
	valRMSE_list = []
	for step in step_array:
		model = LinearRegressionWithSGD.train(train_featureScoreTimeRDD, iterations=5000, step=step)
		labelsAndPreds = val_featureScoreTimeRDD.map(lambda p: (p.label, model.predict(p.features)))
		valMSE = labelsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / val_featureScoreTimeRDD.count()
		valRMSE=valMSE**0.5
		valRMSE_list.append((step, valRMSE))
	return valRMSE_list

steps = [0.01,0.1,1.0,1.5,2.0, 3.0, 5.0, 10.0, 15.0]
#steps = [0.01, 0.1,1.0]

valRMSE_List =  getRMSE(steps)
print valRMSE_List
