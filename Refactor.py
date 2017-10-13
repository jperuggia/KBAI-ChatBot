
"""
Neural network testing doesnt work when training is off, cant figure out why 
"""

"""
Chatbot class
This is the entry point and exit point for your chatbot.
Do not change this API. If it it changes your chatbot will
not be compatible with the autograder.

I highly recommend just calling your code from this file
(put your chatbot code in another file) in case we need to
change this file during the project.
"""
import datetime
import json

import nltk
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import numpy as np
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



class Chatbot:

    def __init__(self,FAQPathFilename):

        TRAIN_NETWORK = True
        TRAINING_FILE_NAME = "synapses.json"

        # FAQPathFilename is string containing
        # path and filename to text corpus in FAQ format.
        self.FAQPathFilename = FAQPathFilename

        self.ERROR_THRESHOLD = 0.2

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

        # network variables
        self.hidden_neurons = 20
        self.alpha = 0.1
        self.iterations = 100000
        self.dropout = False
        self.dropout_percent = 0.5

        self.parse_corpus()  # always parse the corpus.

        self.create_training_data()
        X = np.array(self.training)
        y = np.array(self.output)

        if TRAIN_NETWORK:
            self.train_network(X, y)
            # test the classify right after training.
            self.classify("Who is professor Goel")
        # else:  # read in the training set that was generated.

        # reset the synapse to read from the file.
        with open(TRAINING_FILE_NAME) as data_file:
            loaded_data = json.load(data_file)
        self.synapse_file_0 = np.asarray(loaded_data['synapse0'])
        self.synapse_file_1 = np.asarray(loaded_data['synapse1'])

        # test after reset.
        self.classify("Who is professor Goel", True)

        breakpoint_garbage = 1234

    def parse_corpus(self):

        with open(self.FAQPathFilename, "r", encoding="utf-8") as f:
            # generate a training set from the corpus. This will be used
            # only if doing a training of the Neural network.
            FAQasList = f.readlines()
            training_set = []
        for s in FAQasList:
            question = s.split("?")[0]
            answer = s.split("?")[1].rstrip()  #remote the new line char too.
            training_set.append({"answer": answer, "question": question})

        # loop over each sentence in training data.
        for pattern in training_set:
            w = nltk.word_tokenize(pattern["question"])
            self.words.extend(w)
            self.documents.append((w, pattern["answer"]))
            # add class to list if not already there.
            if pattern["answer"] not in self.classes:
                self.classes.append(pattern["answer"])

        # stem and lower each word, remove duplicates.
        ignore_words = ["?", "!", "."]
        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]
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

        now = datetime.datetime.now()

        # persist synapses
        synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                   'datetime': now.strftime("%Y-%m-%d %H:%M"),
                   'words': self.words,
                   'classes': self.classes
                   }
        synapse_file = "synapses.json"

        # set the global self to the things that are needed for synapse.
        self.synapse_1 = np.asarray(synapse_1)
        self.synapse_0 = np.asarray(synapse_0)

        # dump the results into a file to be read in.
        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)

        print("saved training results to:", synapse_file)

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
        print("%s \n classification: %s" % (sentence, return_results))
        return return_results

    def UserFeedback(self,yesorno):
        #TODO: user calls this with "yes" or "no" feedback when InputOutput returns TRUE
        return

    def InputOutput(self,msg):
        # msg is text to chatbot: question or "yes" or "no"
        # return expect response from user, agent response
        # return True,  response text as string
        # return False, "I do not know"

        if msg == "Who are you?":
            return False, "KBAI student, " + self.FAQPathFilename

        # TODO: Insert calls to your chatbot here
        #       Your chatbot should return '' if
        #       it does not have an answer.
        response = ''
        for qa in self.FAQasList:           # Example code
            question = qa.split('?')[0]     # Example code
            answer =qa.split('?')[1]        # Example code
            if question == msg:
                response = answer
                break


        # You should not need to change any of the code below
        # this line.

        # If your agent does not know the answer
        if not response:
            return False,"I do not know."

        # If your agent knows the answer
        # True indicates your agent is expecting a "yes" or "no" from the user
        # in the next call to Chatbot()
        # Do not change this return statement
        return True, response + "\nIs the response correct (yes/no)?"

