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
from pyspark.mllib.tree import RandomForest, RandomForestModel


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

def getRandomForestRMSE(trees_array):
	valRMSE_list = []
	for trees in trees_array:
		model = RandomForest.trainRegressor(train_featureScoreTimeRDD, categoricalFeaturesInfo={},
                                    numTrees=trees, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)
		predictions = model.predict(val_featureScoreTimeRDD.map(lambda lp: lp.features))
		labelsAndPreds = val_featureScoreTimeRDD.map(lambda lp: lp.label).zip(predictions)
		valMSE = labelsAndPreds.map(lambda (v, p): (v - p)*(v-p)).sum() / float(val_featureScoreTimeRDD.count())
		valRMSE=valMSE**0.5
		valRMSE_list.append((trees, valRMSE))
	return valRMSE_list

trees_array = [10, 50, 100, 500, 1000]

valRMSE_List =  getRandomForestRMSE(trees_array)

print "RMSE"
print valRMSE_List
