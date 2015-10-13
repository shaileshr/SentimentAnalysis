import re
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel
import datetime
import numpy as np
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from collections import defaultdict
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import Normalizer

path = 'Lab2/Assignment3'

conf = SparkConf()

conf.setMaster('yarn-client')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

df = sqlContext.read.json(path+'/reviews_Pet_Supplies_p1.json')

reviewDF = df.select("overall", "reviewText", "reviewTime")

def removePunctuation(text):
   return re.sub("[^a-zA-Z]", " ", text)

cleanedReviewRDD = reviewDF.map(lambda row: (row.overall, removePunctuation(row.reviewText).lower().split(), row.reviewTime ))

wordsClustersRDD= sc.pickleFile(path+'/WordClustersRDD',10)

reviewRDDWithIndex = cleanedReviewRDD.zipWithIndex().map(lambda (row, index):(index, row)).cache()
reviewTextWithIndex = reviewRDDWithIndex.map(lambda (index,(score, words, time)): (index, words))
reviewScoreTimeWithIndex = reviewRDDWithIndex.map(lambda (index,(score, words, time)): (index, (score, time)))

def getKey(item):
    return item[0]


def createSparseVector(histogram):
	indexList = []
	countList = []
	for histogramIndex, count in sorted(histogram, key=getKey):
		indexList.append(histogramIndex)
		countList.append(count)
	return Vectors.sparse(2000, indexList,countList)


reviewFeaturesRDD = reviewTextWithIndex\
	.flatMapValues(lambda word: word)\
	.map(lambda (rowNum, word): (word, rowNum))\
	.join(wordsClustersRDD)\
	.map(lambda (word, (rowNum,clusterIndex)): ((rowNum, clusterIndex), 1))\
	.reduceByKey(lambda x,y: x+y)\
	.map(lambda ((rowNum, clusterIndex), count):(rowNum, (clusterIndex, count)))\
	.groupByKey()\
	.mapValues(lambda histogram: createSparseVector(histogram))\
	.cache()


norm = Normalizer(1)

normalisedReviewFeaturesRDD=reviewFeaturesRDD.map(lambda (rowNum, features):rowNum)\
	.zip(norm.transform(reviewFeaturesRDD.map(lambda (rowNum, features):features)))

formatter_string = "%m %d %Y"

allDataRDD=normalisedReviewFeaturesRDD.join(reviewScoreTimeWithIndex)\
	.map(lambda (rowNum, (features, (score, time))):(score, features, datetime.datetime.strptime(time[-4:], "%Y").year))\
	.cache()

train_featureScoreTimeRDD=allDataRDD.filter(lambda (score, feature, time): time<2014 )\
	.map(lambda (score, feature, time): LabeledPoint(float(score),feature))\
	.repartition(10).cache()
val_featureScoreTimeRDD=allDataRDD.filter(lambda (score, feature, time): time>=2014 )\
	.map(lambda (score, feature, time): LabeledPoint(float(score),feature))\
	.repartition(10).cache()
allDataRDD.unpersist()
reviewFeaturesRDD.unpersist()

def getRMSE(step_array):
	valRMSE_list = []
	for step in step_array:
		model = LinearRegressionWithSGD.train(train_featureScoreTimeRDD, iterations=5000, step=step)
		labelsAndPreds = val_featureScoreTimeRDD.map(lambda p: (p.label, model.predict(p.features)))
		valMSE = labelsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / val_featureScoreTimeRDD.count()
		valRMSE=valMSE**0.5
		valRMSE_list.append((step, valRMSE))
	return valRMSE_list

steps = [0.01,0.1,1.0,1.5,2.0, 3.0, 5.0, 10.0, 15.0, 50.0,100.0, 500.0,1000.0]

valRMSE_List =  getRMSE(steps)

print "RMSE"
print valRMSE_List
