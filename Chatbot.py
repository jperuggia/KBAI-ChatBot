"""
Chatbot class
This is the entry point and exit point for your chatbot.
Do not change this API. If it it changes your chatbot will
not be compatible with the autograder.

I highly recommend just calling your code from this file
(put your chatbot code in another file) in case we need to
change this file during the project.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import time


stemmer = nltk.LancasterStemmer()
lamma = WordNetLemmatizer()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_output_to_derivative(output):
    return output*(1-output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [lamma.lemmatize(w.lower()) for w in sentence_words]
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


# Method to determine if the sentence contains
# an Assignment # or Project # word. To help with accuracy
# the bot will combine these to be a single word.
def detect_project_or_assignment(sentence):
    sentence = sentence.lower()
    a = re.search(r'\b(project)\b\s\d+', sentence)
    p = re.search(r'\b(assignment)\b\s\d+', sentence)

    if a is not None and a.start() >=0:
        p1 = sentence[:a.start()]
        p2 = sentence[a.start(): a.end()]
        p2 = "".join(p2.split())
        p3 = sentence[a.end():]
        sentence = p1 + p2 + p3
    elif p is not None and p.start() >= 0:
        p1 = sentence[:p.start()]
        p2 = sentence[p.start(): p.end()]
        p2 = "".join(p2.split())
        p3 = sentence[p.end():]
        sentence = p1 + p2 + p3

    return sentence


# inspiration for NN design can be found here.
# https://iamtrask.github.io/2015/07/12/basic-python-network/
# https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6


class Chatbot:

    def __init__(self,FAQPathFilename):
        TRAIN_NETWORK = True

        # FAQPathFilename is string containing
        # path and filename to text corpus in FAQ format.
        self.FAQPathFilename = FAQPathFilename
        self.ERROR_THRESHOLD = 0.15

        # our learned data set. The black box of NO!
        self.QUESTION_ASKED = ""
        self.Incorrect_Answers = []

        # The neural network will use these to classify inputs and outputs.
        self.documents = []
        self.classes = []
        self.words = []
        self.ignore_words = ["?", ".", "!", ",", "'s", "2"]
        self.s_words = set(stopwords.words('english'))

        # the training data for the NN
        self.training = []
        self.output = []

        # network layers.
        self.synapse_0 = []
        self.synapse_1 = []

        # network variables
        self.hidden_neurons = 11
        self.alpha = 0.1
        self.iterations = 50000
        self.parse_corpus()  # always parse the corpus.

        self.create_training_data()
        X = np.array(self.training)
        y = np.array(self.output)

        if TRAIN_NETWORK:
            self.train_network(X, y)

    def parse_corpus(self):

        with open(self.FAQPathFilename, "r", encoding="utf-8") as f:
            # generate a training set from the corpus. This will be used
            # only if doing a training of the Neural network.
            FAQasList = f.readlines()

        training_set = []

        for s in FAQasList:
            question = s.split("?")[0]
            answer = s.split("?")[1].rstrip()
            training_set.append({"answer": answer, "question": question.lower()})

        # loop over each sentence in training data.
        for pattern in training_set:
            pattern["question"] = detect_project_or_assignment(pattern["question"])
            w = nltk.word_tokenize(pattern["question"])
            self.words.extend(w)
            self.documents.append((w, pattern["answer"]))
            # add class to list if not already there.
            if pattern["answer"] not in self.classes:
                self.classes.append(pattern["answer"])

        # stem and lower each word, remove duplicates.
        # remove stop words?
        self.words = [w for w in self.words if w not in self.s_words]
        self.words = [lamma.lemmatize(w) for w in self.words]
        self.words = [stemmer.stem(w) for w in self.words]
        self.words = [w.lower() for w in self.words if w not in self.ignore_words]
        self.words = list(set(self.words))
        # sort it!
        self.words = sorted(self.words)

        self.classes = list(set(self.classes))

    def create_training_data(self):
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]

            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            self.training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            self.output.append(output_row)

    def train_network(self, X, y):
        print("The chatbot is learning.. please be patient")
        np.random.seed(1)
        last_mean_error = 1

        s0 = 2 * np.random.random((len(X[0]), self.hidden_neurons)) - 1
        s1 = 2 * np.random.random((self.hidden_neurons, len(self.classes))) - 1
        prev_synapse_0_weight_update = np.zeros_like(s0)
        prev_synapse_1_weight_update = np.zeros_like(s1)

        synapse_0_direction_count = np.zeros_like(s0)
        synapse_1_direction_count = np.zeros_like(s1)

        for j in iter(range(self.iterations + 1)):
            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(np.dot(layer_0, s0))
            layer_2 = sigmoid(np.dot(layer_1, s1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j % 10000) == 0 and j > 5000:

                # if the error is getting worse stop.
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    last_mean_error = np.mean(np.abs(layer_2_error))

            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(s1.T)
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if j > 0:
                synapse_0_direction_count += np.abs(
                    ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(
                    ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

            s1 += self.alpha * synapse_1_weight_update
            s0 += self.alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        # set the global self to the things that are needed for synapse.
        self.synapse_1 = np.asarray(s1)
        self.synapse_0 = np.asarray(s0)

    def think(self, sentence):
        # clean up the question into usable data.
        sentence = detect_project_or_assignment(sentence)
        x = bag_of_words(sentence, self.words)

        level_0 = x
        level_1 = sigmoid(np.dot(level_0, self.synapse_0))
        level_2 = sigmoid(np.dot(level_1, self.synapse_1))

        return level_2

    def classify(self, sentence):
        results = self.think(sentence)
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results = [[self.classes[r[0]], r[1]] for r in results]
        return return_results

    def UserFeedback(self,yesorno):
        if yesorno.lower() == 'no':
            # the answer we gave is not correct. Save this question to the black box of no.
            self.Incorrect_Answers.append(self.QUESTION_ASKED)
            self.QUESTION_ASKED = ""

            #### Failed code at trying to retrain NN.
            # q_fix = detect_project_or_assignment(self.QUESTION_ASKED.lower())
            # new_data_set = {"answer": "I do not know", "question": q_fix}
            # words = nltk.word_tokenize(new_data_set["question"])
            # w = [w for w in words if w not in self.s_words]
            # w = [lamma.lemmatize(w) for w in w]
            # w = [stemmer.stem(w) for w in w]
            # self.words.extend(w)
            # self.words = list(set(self.words))
            # self.words = sorted(self.words)
            # if "I do not know" not in self.classes:
            #     self.classes.append("I do not know")
            # # remove all old knowledge from data.
            # self.training = []
            # self.output = []
            # # recreate
            # self.create_training_data()
            # # retrain.
            # X = np.array(self.training)
            # y = np.array(self.output)
            # self.train_network(X,y)
        return

    def InputOutput(self,msg):

        if msg == "Who are you?":
            return False, "KBAI student, " + self.FAQPathFilename

        # save the question off in the memory for learning based on feedback.
        self.QUESTION_ASKED = msg.lower()

        # if the question was asked and is in the blackbox. DO NOT USE IT!
        if self.QUESTION_ASKED in self.Incorrect_Answers:
            return False, "I do not know."

        r = self.classify(msg)
        # get the best value from R that I can find.
        if len(r) < 1:
            response = ''
        else:
            response = max(r, key=lambda item: item[1])[0]

        # If your agent does not know the answer
        if not response:
            return False,"I do not know."

        # If your agent knows the answer
        # True indicates your agent is expecting a "yes" or "no" from the user
        # in the next call to Chatbot()
        # Do not change this return statement
        return True, response + "\nIs the response correct (yes/no)?"

