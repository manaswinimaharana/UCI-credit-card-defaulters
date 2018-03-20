#The first step is to initialize sparkContext

from pyspark.sql import SparkSession

spark = SparkSession.builder \
      .appName("Intro to Spark") \
      .getOrCreate()
sc=spark.sparkContext

from __future__ import print_function
import sys

inputDF = spark   \
      .read   \
      .format("csv") \
      .option("header", True) \
      .option("inferSchema", True) \
      .load("/tmp/credit_data_input/UCI_Credit_Card.csv")

inputDF.printSchema
inputDF.dtypes
inputDF.coalesce(1)
from pyspark.sql.functions import *
#let's rename input column and change ID data type to String

#In order to transform let's register the table
inputDF.createOrReplaceTempView("inputDF")
spark.sql("Select cast(ID as String) as ID, \
LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0, \
PAY_2,PAY_3,PAY_4,PAY_5 ,\
PAY_6,BILL_AMT1,BILL_AMT2, \
BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6, \
PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,\
PAY_AMT6, `default.payment.next.month` as Y from inputDF").createOrReplaceTempView("trans_df1")


spark.sql("select Y,count(*) from inputDF group by Y ").show()
#Dataset is inbalances let's create some weights 
tblSize=spark.sql("Select * from trans_df1")
datasetSize=tblSize.count()

yNegatives=spark.sql("Select 1 from trans_df1 where Y<0")
numNegatives=yNegatives.count()


balancingRatio=(datasetSize - numNegatives)/datasetSize
trans_df3=spark.sql("select * from trans_df1")
trans_df3=trans_df2.withColumn('classWeightCol', when(trans_df2.Y == 0.0,balancingRatio).otherwise(1.0 - balancingRatio ))
trans_df3.show(3)


#Encoding Categorical Variable 
from pyspark.sql.functions import *
categoricalCol=["SEX","MARRIAGE","AGE","EDUCATION","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

for c in categoricalCol:
  str1=c+"_Index"
  str2=c+"_Vec"
  stringIndexer=StringIndexer().setInputCol(c).setOutputCol(str1)
  model = stringIndexer.fit(trans_df3)
  
  indexed = model.transform(trans_df3)
  encoder= OneHotEncoder().setInputCol(str1).setOutputCol(str2)
  trans_df3=encoder.transform(indexed)

trans_df3.show(3)


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#assembler = VectorAssembler(
#    inputCols=["LIMIT_BAL", "SEX", "EDUCATION","MARRIAGE","AGE"],
#    outputCol="features")

assembler=VectorAssembler(inputCols=["LIMIT_BAL","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6", \
                                    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","SEX_Vec","MARRIAGE_Vec", \
                                    "AGE_Vec","EDUCATION_Vec","PAY_0_Vec","PAY_2_Vec","PAY_3_Vec","PAY_4_Vec","PAY_5_Vec","PAY_6_Vec"],outputCol="features")

output = assembler.transform(trans_df3)


#Split the training & test data 
(trainingData, testData) = output.randomSplit([0.7, 0.3])

from pyspark.ml.evaluation import BinaryClassificationEvaluator
binaryEvaluator=BinaryClassificationEvaluator(labelCol="Y",rawPredictionCol="rawPrediction")
binaryEvaluator.setMetricName("areaUnderROC")

from pyspark.ml.evaluation import RegressionEvaluator
evaluatorRegression=RegressionEvaluator(labelCol="Y",predictionCol="prediction")
evaluatorRegression.setMetricName("rmse")

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol='Y',maxIter=10, regParam=0.03, elasticNetParam=0.8)
model = lr.fit(trainingData)

print(model.summary.areaUnderROC)

prediction=model.transform(trainingData)
areaTraining=binaryEvaluator.evaluate(prediction)
print("Area Under ROC using Logistics Regression on training data =" + str(areaTraining))
predictionTest=model.transform(testData)
areaTest=binaryEvaluator.evaluate(predictionTest)
print("Area Under ROC using Logistics Regression on test data =" + str(areaTest))
rmseLR = evaluatorRegression.evaluate(predictionTest)
print("Root mean squared error using Logistics Regression on test data =" + str(rmseLR))




print(model.coefficientMatrix)
print(model.summary.areaUnderROC)
print(model.numFeatures)
print(model.coefficients)
print(model.summary.objectiveHistory)
print(model.summary.featuresCol)
print(model.summary.predictions).select("rawPrediction","probability","prediction").where("ID=10").show(2)
print(model.summary.probabilityCol)



model.coefficientMatrix.values


model.summary.predictions.select("rawPrediction","prediction","features").show()
model.summary.predictions.show(1)
#lr = LogisticRegression(labelCol='Y',maxIter=10, regParam=0.3, elasticNetParam=0.8)
#Area Under ROC using Logistics Regression on training data =0.5
#Area Under ROC using Logistics Regression on test data =0.5
#Root mean squared error using Logistics Regression on test data =0.472435684741
#

#HYPER-PARAMETER-TUNING

lr = LogisticRegression(labelCol='Y',maxIter=20, regParam=0.03, elasticNetParam=0.8)
model = lr.fit(trainingData)
#print(model.summary.areaUnderROC)
#0.760107793348


lr = LogisticRegression(labelCol='Y',maxIter=50, regParam=0.03, elasticNetParam=0.8)
model = lr.fit(trainingData)
#print(model.summary.areaUnderROC)
#0.760454440376

lr = LogisticRegression(labelCol='Y',maxIter=50, regParam=0.07, elasticNetParam=0.8)
model = lr.fit(trainingData)
#print(model.summary.areaUnderROC)
#0.716606411216



lr = LogisticRegression(labelCol='Y',maxIter=50, regParam=0.01, elasticNetParam=0.8)
model = lr.fit(trainingData)
#print(model.summary.areaUnderROC)
#0.716606411216


#After Iterations let's persist/save models 
model.save("/tmp/ml_ouput/test_model")

import org.apache.spark.ml.regression.LinearRegressionModel
#Load the fitted model back 
sameCVModel = LinearRegressionModel.load("/tmp/ml_ouput/test_model")




# To VISUALIZE 

from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
results = predictionTest.select(['probability', 'Y'])
## prepare score-label set
results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)
metrics = metric(scoreAndLabels)
print("The ROC score is (@numTrees=200): ", metrics.areaUnderROC)


from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
 
y_test = [i[1] for i in results_list]
y_score = [i[0] for i in results_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

from pylab import *
ioff()

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#To gain a better understanding of our model’s performance, 
#we can plot the distribution of our predictions:
all_probs = predictionTest.select("probability").collect()
pos_probs = [i[0][0] for i in all_probs]
neg_probs = [i[0][1] for i in all_probs]
 
# pos
plt.hist(pos_probs, 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('predicted_values')
plt.ylabel('Counts')
plt.title('Probabilities for positive cases')
plt.grid(True)
plt.show()
 
# neg
plt.hist(neg_probs, 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('predicted_values')
plt.ylabel('Counts')
plt.title('Probabilities for negative cases')
plt.grid(True)
plt.show()