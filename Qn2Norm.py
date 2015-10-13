import re
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf

import nltk
from nltk.corpus import stopwords
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import datetime
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.feature import Normalizer


conf = SparkConf()

conf.setMaster('yarn-client')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


path = "/Users/sradhakr/Desktop/Assignment3/Assignment3"

train_featureScoreTimeRDD=sc.pickleFile(path+'trainDataRDD',10)
val_featureScoreTimeRDD=sc.pickleFile(path+'valDataRDD',10)

norm = Normalizer(2)



train_featureScoreTimeRDD=sc.pickleFile(path+'trainDataRDD',10)
val_featureScoreTimeRDD=sc.pickleFile(path+'valDataRDD',10)


train_featuresRDD=train_featureScoreTimeRDD.map(lambda (feature, score): feature)

trainfeatureScoreNormRDD=norm.transform(train_featuresRDD).zip(train_featuresRDD.map(lambda (feature, score): score))


val_featuresRDD=val_featureScoreTimeRDD.map(lambda (feature, score): feature)

valfeatureScoreNormRDD=norm.transform(val_featuresRDD).zip(val_featuresRDD.map(lambda (feature, score): score))


train_data = trainfeatureScoreNormRDD.map(lambda (feature, score): LabeledPoint(float(score), feature)).repartition(10).cache()
val_data = valfeatureScoreNormRDD.map(lambda (feature, score): LabeledPoint(float(score), feature)).repartition(10).cache()


def getRMSE(step_array):
	valRMSE_list = []
	for step in step_array:
		model = LinearRegressionWithSGD.train(train_data, iterations=5000, step=step)
		labelsAndPreds = val_data.map(lambda p: (p.label, model.predict(p.features)))
		valMSE = labelsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / val_data.count()
		valRMSE=valMSE**0.5
		valRMSE_list.append((step, valRMSE))
	return valRMSE_list

steps = [0.01,0.1,1.0,1.5,2.0]

valRMSE_List =  getRMSE(steps)

train_data.unpersist()
val_data.unpersist()

print valRMSE_List