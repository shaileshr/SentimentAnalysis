import re
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from nltk.corpus import stopwords 

import nltk
from nltk.corpus import stopwords
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import datetime
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

conf = SparkConf()

conf.setMaster('yarn-client')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


path = "/Users/sradhakr/Desktop/Assignment3/Assignment3"

train_featureScoreTimeRDD=sc.pickleFile(path+'trainDataRDD',10)
val_featureScoreTimeRDD=sc.pickleFile(path+'valDataRDD',10)

train_data = train_featureScoreTimeRDD.map(lambda (feature, score): LabeledPoint(float(score), feature)).repartition(10).cache()
val_data = val_featureScoreTimeRDD.map(lambda (feature, score): LabeledPoint(float(score), feature)).repartition(10).cache()


def getRMSE(step_array):
	valRMSE_list = []
	for step in step_array:
		model = LinearRegressionWithSGD.train(train_data, iterations=1000, step=step)

		labelsAndPreds = val_data.map(lambda p: (p.label, model.predict(p.features)))
		valMSE = labelsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / val_data.count()
		valRMSE=valMSE**0.5
		valRMSE_list.append((step, valRMSE))
	return valRMSE_list
	
steps = [0.01,0.1,1.0,10.0,100.0]		
	
valRMSE_List =  getRMSE(steps)

train_featureScoreTimeRDD.unpersist()
val_featureScoreTimeRDD.unpersist()
train_data.unpersist()
val_data.unpersist()

print valRMSE_List