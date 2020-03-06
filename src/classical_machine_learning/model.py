import io
import re
import numpy as np#支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
#泛用机器学习框架
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
#CountVectorizer 类会将文本中的词语转换为词频矩阵，进行词频向量化。TfidfTransformer是进行tf-idf预处理。
from sklearn import svm#使用svm模型
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier#集成分类器
from sklearn.linear_model import LogisticRegression#逻辑回归
import time
import os
#引入数据集中的训练集和测试集以及特征向量
gloveDir = "../../data"
ssweDir = "C://Users//hp//emocontext//sswe"
trainDataPath = "../../data/train.txt"
testDataPath = "../../data/test.txt"
solutionPath = "../../data/solution.txt"
featureVectorsPath = "./temp/features.npy"
featureVectorsPath_2 = "./temp/features2.npy"
#定义标签
label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:#用符号将数据集分成多个数组
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')#去掉空串
                    except:
                        break
                cSpace = ' ' + c + ' '
                #print('cSpace为：',cSpace)
                line = cSpace.join(lineSplit)
                #print(line)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                #print('此处的label为：',label)
                labels.append(label)

            # conv = ' <eos> '.join(line[1:4])
            conv = ' '.join(line[1:4])
            #print('conv1为：',conv)
            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            #print('line为：',line)
            #print('conv2为：',conv)
            indices.append(int(line[0]))
            #print(indices)
            #print(int(line[0]))
            conversations.append(conv.lower())
            #print(conv.lower())
    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def extract_features(corpus):#用tf-idf的方式获得每组对话的特征向量
    """
    :param corpus: list of conversations (train + test)
    :return: list of feature vectors (a vector for each conversation)
    """
    vectorizer = CountVectorizer()
    #print('train + test:',corpus)
    count = vectorizer.fit_transform(corpus)

    transformer = TfidfTransformer(smooth_idf=True)
    feature_vectors = transformer.fit_transform(count)#获取tf-idf值
    #print('tfidf:',feature_vectors)
    return feature_vectors#获取tf-idf特征向量


def extract_features_glove(corpus, load=False):#使用全局词频统计工具完成词向量的转化
    if load:
        featureVectors = np.load(featureVectorsPath)
        return featureVectors

    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.twitter.27B.100d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    featureVectors = np.zeros((len(corpus), 100))
    tfidf = getTfidf(corpus)

    for s in range(len(corpus)):
        sentence = corpus[s].split(" ")
        sentence2vec = np.zeros(100)
        for word in sentence:
            embeddingVector = embeddingsIndex.get(word)
            # words not found in embedding index will be all-zeros
            if embeddingVector is not None:
                sentence2vec += (tfidf.get(word, 0.0) * embeddingVector)

        featureVectors[s] = sentence2vec

    np.save(featureVectorsPath, featureVectors)
    #print('GloVe:',featureVectors)
    return featureVectors

def getTfidf(corpus, load=False):
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)

    # map words to tf-idf scores
    keys = vectorizer.get_feature_names()
    values = vectorizer.idf_
    tfidf = dict(zip(keys, values))
    #print('tfidf为：',tfidf)
    return tfidf


def build_model(name="svm"):#根据不同选择选出模型

    model_1 = svm.SVC(gamma='scale', verbose=True)
    model_2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=7)
    model_3 = LogisticRegression(multi_class="auto", solver="liblinear", random_state=7, verbose=True)
    model = VotingClassifier(estimators=[('lr', model_3), ('gbc', model_2), ('svm', model_1)], voting='hard')

    name2model = {"svm": model_1, "gbc": model_2, "lr": model_3, "ensemble": model}
    print('使用分类器为：',name)
    return name2model[name]

EMOS = ['happy', 'angry', 'sad', 'others']
EMOS_DIC = {'happy': 0,
            'angry': 1,
            'sad': 2,
            'others': 3}
NUM_EMO = len(EMOS)

def to_categorical(vec):
    vec = np.asarray(vec)
    print('类型vec：', type(vec) )
    to_ret = np.zeros(( vec.shape[0], NUM_EMO))  # vec.shape[0]   numpy
    for idx, val in enumerate(vec):
        to_ret[idx, val] = 1
    return to_ret

import statistics as s
def get_metrics(ground, predictions):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref -
        https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions)
    ground = to_categorical(ground)
    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    #  Macro level calculation
    macroPrecision = 0
    macroRecall = 0
    f1_list = []
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(NUM_EMO-1):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (EMOS[c], precision, recall, f1))
    print('Harmonic Mean: ',
          s.harmonic_mean(f1_list))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) \
        if (macroPrecision + macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
    macroPrecision, macroRecall, macroF1))

    # Micro level calculation
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d"
          % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall)\
        if (microPrecision + microRecall) > 0 else 0

    # predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
    accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1

def main():
    start_time = time.time()

    balanced = False
    indices_train, conversations_train, labels_train = preprocessData(trainDataPath, mode="train")
    if balanced:
        conversations_train = conversations_train[:len(conversations_train)//2]
        labels_train = labels_train[:len(labels_train)//2]
    indices_test, conversations_test, labels_test = preprocessData(testDataPath, mode="train")

    corpus = conversations_train + conversations_test

    print("Training data size - %d" % len(conversations_train))
    print("Test data size - %d" % len(conversations_test))


    feature_vectors = extract_features(corpus)

    x_train = feature_vectors[:len(conversations_train)]
    x_test = feature_vectors[len(conversations_train):]
    print("Number of features - %d" % x_train.shape[1])

    print("Preparing training...")
    y_train = np.array(labels_train)
    #print(y_train)
    model = build_model(name="lr")
    model.fit(x_train, y_train)
    print("Training accuracy - %.2f" % model.score(x_train, y_train))

    print("Preparing testing...")
    predictions = model.predict(x_test)

    print(' 结果---')
    get_metrics(labels_test,predictions )

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Training and testing completed.")

    print("Time taken for training and testing - %.2fs" % (time.time() - start_time))


if __name__ == "__main__":
    main()