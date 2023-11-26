import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,  IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer

spark = SparkSession.builder.appName("EnglishWithBanglaLanguageProcessing").getOrCreate()

txt_data = spark.read.text("data.txt")
txt_data.show()

def sentence_pos_tag(sentence):
    tokens = word_tokenize(sentence)

    tokens_without_punctuation = [token for token in tokens if token.isalnum()]

    pos_tags = pos_tag(tokens_without_punctuation)

    pos_tagged_sentence = " ".join(tag for word, tag in pos_tags)
    return pos_tagged_sentence

udf_sentence_pos_tag = udf(sentence_pos_tag, StringType())

def generate_problems(row):
    words = row['pos_tags'].split()

    variations = []

    for i in range(len(words)):
        placeholder_sentence = " ".join(["___" if j == i else word for j, word in enumerate(words)])
        variations.append((placeholder_sentence, words[i]))

    return variations

txt_data = txt_data.withColumn("pos_tags", udf_sentence_pos_tag(col("value")))
txt_data.show()

new_txt_data = txt_data.rdd.flatMap(generate_problems).collect()
columns = ["Question", "Answer"]
df = spark.createDataFrame(new_txt_data, columns)
df.show()

label_indexer = StringIndexer(inputCol="Answer", outputCol="Answer_index")
li_model = label_indexer.fit(df)
df = li_model.transform(df)
df.show()

train_data, test_data = df.randomSplit([0.8, 0.2], seed=3)
train_data.show()
test_data.show()

tokenizer = Tokenizer(inputCol="Question", outputCol="Tokenized_Question")

word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="Tokenized_Question", outputCol="Train_Question")

index_to_string = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=li_model.labels)

rf = RandomForestClassifier(labelCol="Answer_index", featuresCol="Train_Question", numTrees=100)

pipeline = Pipeline(stages=[tokenizer, word2Vec, rf, index_to_string])

model = pipeline.fit(train_data)

result_df = model.transform(test_data)

result_df.select("Question", "Answer", "predicted_label").show()

evaluator = MulticlassClassificationEvaluator(labelCol="Answer_index", predictionCol="prediction", metricName="accuracy")

predictionAndLabels = result_df.select("prediction", "Answer_index").rdd.map(lambda x: (float(x.prediction), float(x.Answer_index)))

metrics = MulticlassMetrics(predictionAndLabels)

print("Confusion Matrix:")
print(metrics.confusionMatrix().toArray())

precision = metrics.precision(label=1.0)
recall = metrics.recall(label=1.0)
f1_score_manual = 2 * (precision * recall) / (precision + recall)
accuracy = evaluator.evaluate(result_df)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score_manual}")

categories = ["Random Forest"]
values1 = [accuracy]
values2 = [precision]
values3 = [recall]
values4 = [f1_score_manual]

vectorizer = CountVectorizer(inputCol="Tokenized_Question", outputCol="Train_Question")

nb = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol="Train_Question", labelCol="Answer_index")

pipeline = Pipeline(stages=[tokenizer, vectorizer, nb, index_to_string])

model = pipeline.fit(train_data)

result_df = model.transform(test_data)

result_df.select("Question", "Answer", "predicted_label").show()

evaluator = MulticlassClassificationEvaluator(labelCol="Answer_index", predictionCol="prediction", metricName="accuracy")

predictionAndLabels = result_df.select("prediction", "Answer_index").rdd.map(lambda x: (float(x.prediction), float(x.Answer_index)))

metrics = MulticlassMetrics(predictionAndLabels)

print("Confusion Matrix:")
print(metrics.confusionMatrix().toArray())

precision = metrics.precision(label=1.0)
recall = metrics.recall(label=1.0)
f1_score_manual = 2 * (precision * recall) / (precision + recall)
accuracy = evaluator.evaluate(result_df)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score_manual}")

categories.append("Naive Bayes")
values1.append(accuracy)
values2.append(precision)
values3.append(recall)
values4.append(f1_score_manual)

lr = LogisticRegression(featuresCol="Train_Question", labelCol="Answer_index")

pipeline = Pipeline(stages=[tokenizer, word2Vec, lr, index_to_string])

model = pipeline.fit(train_data)

result_df = model.transform(test_data)

result_df.select("Question", "Answer", "predicted_label").show()

evaluator = MulticlassClassificationEvaluator(labelCol="Answer_index", predictionCol="prediction", metricName="accuracy")

predictionAndLabels = result_df.select("prediction", "Answer_index").rdd.map(lambda x: (float(x.prediction), float(x.Answer_index)))

metrics = MulticlassMetrics(predictionAndLabels)

print("Confusion Matrix:")
print(metrics.confusionMatrix().toArray())

precision = metrics.precision(label=1.0)
recall = metrics.recall(label=1.0)
f1_score_manual = 2 * (precision * recall) / (precision + recall)
accuracy = evaluator.evaluate(result_df)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score_manual}")

categories.append("Logistic Regression")
values1.append(accuracy)
values2.append(precision)
values3.append(recall)
values4.append(f1_score_manual)


import matplotlib.pyplot as plt
import numpy as np

bar_width = 0.2

r1 = np.arange(len(categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

plt.bar(r1, values1, color='b', width=bar_width, edgecolor='grey', label='Accuracy')
plt.bar(r2, values2, color='g', width=bar_width, edgecolor='grey', label='Precision')
plt.bar(r3, values3, color='r', width=bar_width, edgecolor='grey', label='Recall')
plt.bar(r4, values4, color='c', width=bar_width, edgecolor='grey', label='F1 Score')

plt.xlabel('Algorithms', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(categories))], categories)

plt.legend()

plt.title('Algorithm output comparison')

plt.show()

