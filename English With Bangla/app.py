from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    data = request.get_json()
    user_input = data['input']

    from nltk.tokenize import sent_tokenize
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize


    user_input_check = sent_tokenize(user_input)
    
    if len(user_input_check) != 1:
        return "Please enter a sentence at a time"

    import re

    ban_w = ""
    ques = ""

    bengali_word_pattern = re.compile(r'[\u0980-\u09FF]+')

    match = re.search(bengali_word_pattern, user_input)

    if match:
        ban_w = match.group()

        ques = re.sub(bengali_word_pattern, '___', user_input)
    else:
        return jsonify(user_input)

    import findspark
    findspark.init()

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StringType

    spark = SparkSession.builder.appName("LanguageBlender").getOrCreate()

    import os
    from pyspark.ml import PipelineModel

    script_directory = os.path.dirname(os.path.abspath(__file__))

    saved_model_directory = os.path.join(script_directory, "model")

    model = PipelineModel.load(saved_model_directory)

    tokens = word_tokenize(ques)
    
    pos_tags = pos_tag(tokens)

    formatted_pos_tags = ' '.join([f'{tag}' if word != "___" else "___" for word, tag in pos_tags])
    
    df = [(formatted_pos_tags,)]
    #print(df)

    new_df = spark.createDataFrame(df , ["Question"])
    result = model.transform(new_df)
    ans = result.first()
    
    final = ans['predicted_label']

    csv_file_path = "B_E_Dict.csv"

    df = spark.read.csv(csv_file_path, inferSchema=True)

    def word_tag(word):
        tokens = word_tokenize(word)
        pos_tags = pos_tag(tokens)

        if pos_tags:
            return pos_tags[0][1]
        else:
            return None

    udf_word_pos_tag = udf(word_tag, StringType())

    new_df = df.withColumn("pos_tagged", udf_word_pos_tag(col("_c1")))
    
    filtered_df = new_df.filter(col("_c0") == ban_w)

    filtered_df2 = filtered_df.filter(col("pos_tagged") == final)

    #filtered_df2.show()
    
    final_answer = ""
    if(filtered_df2.count() > 0):
        final_answer = filtered_df2.first()
        final_answer = final_answer['_c1']
    else:
        final_answer = filtered_df.first()
        
        final_answer = final_answer['_c1']
    
    #print(final_answer)
    
    dash_index = ques.index("___")
    
    if(dash_index == 0):
        output = ques.replace("___", final_answer)
    else:
        output = ques.replace("___", final_answer.lower())

    return jsonify(output)


@app.route('/add_input', methods=['POST'])
def add_input():
    data = request.get_json()
    user_input = data['input']

    from nltk.tokenize import sent_tokenize

    user_input_check = sent_tokenize(user_input)
    
    if len(user_input_check) != 1:
        return "Please enter a sentence at a time"

    import findspark
    findspark.init()

    from pyspark.sql import SparkSession    
    from pyspark.sql.functions import col

    spark = SparkSession.builder.appName("LanguageBlender").getOrCreate()

    df = spark.read.text("data.txt")

    filtered_df = df.filter(col('value').contains(user_input))

    #filtered_df.show()

    if filtered_df.count() > 0:
        output = "Input already exists"
    else:
        with open("data.txt", 'a') as file:
            file.write(f"{user_input}\n")

        output = "Successfully added new input to the dataset"

    spark.stop()

    return jsonify(output)

@app.route('/train_model', methods=['POST'])
def train_model():
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
    from pyspark.ml.feature import Tokenizer, Word2Vec
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer,  IndexToString

    spark = SparkSession.builder.appName("LanguageBlender").getOrCreate()

    txt_data = spark.read.text("data.txt")
    #txt_data.show()

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
    #txt_data.show()

    new_txt_data = txt_data.rdd.flatMap(generate_problems).collect()
    columns = ["Question", "Answer"]
    df = spark.createDataFrame(new_txt_data, columns)
    #df.show()

    label_indexer = StringIndexer(inputCol="Answer", outputCol="Answer_index")
    li_model = label_indexer.fit(df)
    df = li_model.transform(df)
    #df.show()

    tokenizer = Tokenizer(inputCol="Question", outputCol="Tokenized_Question")

    word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="Tokenized_Question", outputCol="Train_Question")

    index_to_string = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=li_model.labels)

    rf = RandomForestClassifier(labelCol="Answer_index", featuresCol="Train_Question", numTrees=100)

    pipeline = Pipeline(stages=[tokenizer, word2Vec, rf, index_to_string])

    model = pipeline.fit(df)

    import os
    script_directory = os.path.dirname(os.path.abspath(__file__))

    script_model_path = os.path.join(script_directory, "model")

    model.write().overwrite().save(script_model_path)

    #result_df = model.transform(df)

    #result_df.select("Question", "Answer", "predicted_label").show()

    spark.stop()

    output = "Successfully trained new model"

    return jsonify({"message": output})


if __name__ == '__main__':
    app.run(debug=True)

