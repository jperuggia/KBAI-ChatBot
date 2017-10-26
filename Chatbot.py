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
import numpy as np
import re
import time


stemmer = nltk.LancasterStemmer()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_output_to_derivative(output):
    return output*(1-output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
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
    a = re.search(r'\b(project)\b\s\d+', sentence)
    p = re.search(r'\b(assignment)\b\s\d+', sentence)

    if a is not None and a.start() >=0:
        p1 = sentence[:a.start()]
        p2 = sentence[a.start(): a.end()]
        p2 = "".join(p2.split())
        p3 = sentence[a.end(): ]
        sentence = p1 + p2 + p3
    elif p is not None and p.start() >= 0:
        p1 = sentence[:p.start()]
        p2 = sentence[p.start(): p.end()]
        p2 = "".join(p2.split())
        p3 = sentence[p.end(): ]
        sentence = p1 + p2 + p3

    return sentence



class Chatbot:

    def __init__(self,FAQPathFilename):

        TRAIN_NETWORK = True
        TRAINING_FILE_NAME = "synapses_2.json"

        # FAQPathFilename is string containing
        # path and filename to text corpus in FAQ format.
        self.FAQPathFilename = FAQPathFilename

        self.ERROR_THRESHOLD = 0.13

        # The neural network will use these to classify inputs and outputs.
        self.documents = []
        self.classes = []
        self.words = []
        self.ignore_words = ["?", ".", "!", ","]
        # the training data for the NN
        self.training = []
        self.output = []
        # network layers.
        self.synapse_0 = []
        self.synapse_1 = []

        self.synapse_file_0 = []
        self.synapse_file_1 = []

        # Nodes | Output list
        '''''''''''''''''''''
        09      | 129.3  / 132
        10      | 
        13 /.13 | 125.9 / 128
        15      |  124 / 126
        20      |  
        25      |
        3
        '''
        # network variables
        self.hidden_neurons = 13
        self.alpha = 0.1
        self.iterations = 120000
        self.dropout = False
        self.dropout_percent = 0.5

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
            training_set.append({"answer": answer, "question": question})

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
        ignore_words = ["?", "!", "."]

        # remove stop words?
        self.words = [w.lower() for w in self.words if w not in ignore_words]
        s_words = set(stopwords.words('english'))
        self.words = [w for w in self.words if w not in s_words]
        self.words = [stemmer.stem(w) for w in self.words if w not in ignore_words]
        self.words = list(set(self.words))

        self.classes = list(set(self.classes))

    def create_training_data(self):
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
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

        synapse_0 = 2 * np.random.random((len(X[0]), self.hidden_neurons)) - 1
        synapse_1 = 2 * np.random.random((self.hidden_neurons, len(self.classes))) - 1
        prev_synapse_0_weight_update = np.zeros_like(synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(synapse_1)

        synapse_0_direction_count = np.zeros_like(synapse_0)
        synapse_1_direction_count = np.zeros_like(synapse_1)

        for j in iter(range(self.iterations + 1)):
            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(np.dot(layer_0, synapse_0))

            if self.dropout:
                layer_1 *= np.random.binomial([np.ones((len(X), self.hidden_neurons))], 1 - self.dropout_percent)[0] * (
                    1.0 / (1 - self.dropout_percent))

            layer_2 = sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j % 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print("Learning step of " + str(j) + ". Has delta error of: " + str(np.mean(np.abs(layer_2_error))))
                    last_mean_error = np.mean(np.abs(layer_2_error))

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if j > 0:
                synapse_0_direction_count += np.abs(
                    ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(
                    ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

            synapse_1 += self.alpha * synapse_1_weight_update
            synapse_0 += self.alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        # set the global self to the things that are needed for synapse.
        self.synapse_1 = np.asarray(synapse_1)
        self.synapse_0 = np.asarray(synapse_0)

    def think(self, sentence, use_alternate_synapse):
        x = bag_of_words(sentence, self.words)
        level_0 = x
        if use_alternate_synapse:
            level_1 = sigmoid(np.dot(level_0, self.synapse_file_0))
            level_2 = sigmoid(np.dot(level_1, self.synapse_file_1))
        else:
            level_1 = sigmoid(np.dot(level_0, self.synapse_0))
            level_2 = sigmoid(np.dot(level_1, self.synapse_1))
        return level_2

    def classify(self, sentence, use_alternate_synapse = False):
        results = self.think(sentence, use_alternate_synapse)
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results = [[self.classes[r[0]], r[1]] for r in results]
        return return_results

    def UserFeedback(self,yesorno):
        #TODO: user calls this with "yes" or "no" feedback when InputOutput returns TRUE
        return

    def InputOutput(self,msg):

        if msg == "Who are you?":
            return False, "KBAI student, " + self.FAQPathFilename

        msg = detect_project_or_assignment(msg)
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

