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


class Chatbot:

    def __init__(self,FAQPathFilename):

        # FAQPathFilename is string containing
        # path and filename to text corpus in FAQ format.
        self.FAQPathFilename = FAQPathFilename
        with open(FAQPathFilename,"r", encoding="utf-8") as f: # Example code
            self.FAQasList = f.readlines()                     # Example code
            f.close()

        # TODO: Open FAQ and parse question,answers
        #       into knowledge base.

        # the knowledge base will be represented as "frames" within the system.
        # each frame will be the question which was read from the Corpus initially.
        knowledge_base = []
        training_set = []

        print("Loading in Knowledge......")
        for s in self.FAQasList:
            if len(s.split("?")) > 2:
                question = s.split("?")[0]
                answer = s.split("?")[1]
            else:
                question, answer = s.split("?")

            answer = answer.rstrip()
            ts_record = {"answer": answer, "question": question}
            training_set.append(ts_record)

        # organize the stuff into docs, classes and words.
        documents = []
        classes = []
        words = []

        for p in training_set:
            # make a token.
            w = TextBlob(p["question"]).tokenize()
            words.extend(w)
            # add documents
            documents.append((w, p["answer"]))
            # add to classes list
            if p["answer"] not in classes:
                classes.append(p["answer"])

        # stem and lower each word, remove duplicates
        stemmer = nltk.LancasterStemmer()
        ignore_words = ["?", "!", "."]
        words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
        words = list(set(words))

        #remove duplicates
        classes = list(set(classes))

        # create new training data.
        training = []
        output = []

        # create an empty array for output.
        output_empty = [0] * len(classes)
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            output.append(output_row)

        print("Loading Complete!!!")

        # sigmoids things
        # compute sigmoid nonlinearity
        def sigmoid(x):
            output = 1 / (1 + np.exp(-x))
            return output

        # convert output of sigmoid function to its derivative
        def sigmoid_output_to_derivative(output):
            return output * (1 - output)

        def clean_up_sentence(sentence):
            # tokenize the pattern
            sentence_words = nltk.word_tokenize(sentence)
            # stem each word
            sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
            return sentence_words

        # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
        def bow(sentence, words, show_details=False):
            # tokenize the pattern
            sentence_words = clean_up_sentence(sentence)
            # bag of words
            bag = [0] * len(words)
            for s in sentence_words:
                for i, w in enumerate(words):
                    if w == s:
                        bag[i] = 1
                        if show_details:
                            print("found in bag: %s" % w)

            return (np.array(bag))

        def think(sentence, show_details=False):
            x = bow(sentence.lower(), words, show_details)
            if show_details:
                print("sentence:", sentence, "\n bow:", x)
            # input layer is our bag of words
            l0 = x
            # matrix multiplication of input and hidden layer
            l1 = sigmoid(np.dot(l0, synapse_0))
            # output layer
            l2 = sigmoid(np.dot(l1, synapse_1))
            return l2

        def train(X, y, hidden_neurons=20, alpha=0.1, epochs=200000, dropout=False, dropout_percent=0.5):
            print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
                hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
            print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
            np.random.seed(1)

            last_mean_error = 1
            # randomly initialize our weights with mean 0
            synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
            synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1

            prev_synapse_0_weight_update = np.zeros_like(synapse_0)
            prev_synapse_1_weight_update = np.zeros_like(synapse_1)

            synapse_0_direction_count = np.zeros_like(synapse_0)
            synapse_1_direction_count = np.zeros_like(synapse_1)

            for j in iter(range(epochs + 1)):

                # Feed forward through layers 0, 1, and 2
                layer_0 = X
                layer_1 = sigmoid(np.dot(layer_0, synapse_0))

                if dropout:
                    layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                        1.0 / (1 - dropout_percent))

                layer_2 = sigmoid(np.dot(layer_1, synapse_1))

                # how much did we miss the target value?
                layer_2_error = y - layer_2

                if (j % 10000) == 0 and j > 5000:
                    # if this 10k iteration's error is greater than the last iteration, break out
                    if np.mean(np.abs(layer_2_error)) < last_mean_error:
                        print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                        last_mean_error = np.mean(np.abs(layer_2_error))
                    else:
                        print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                        break

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

                if (j > 0):
                    synapse_0_direction_count += np.abs(
                        ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
                    synapse_1_direction_count += np.abs(
                        ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

                synapse_1 += alpha * synapse_1_weight_update
                synapse_0 += alpha * synapse_0_weight_update

                prev_synapse_0_weight_update = synapse_0_weight_update
                prev_synapse_1_weight_update = synapse_1_weight_update

            now = datetime.datetime.now()

            # persist synapses
            synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                       'datetime': now.strftime("%Y-%m-%d %H:%M"),
                       'words': words,
                       'classes': classes
                       }
            synapse_file = "synapses.json"

            # dump the results into a file to be read in.
            with open(synapse_file, 'w') as outfile:
                json.dump(synapse, outfile, indent=4, sort_keys=True)

            print("saved synapses to:", synapse_file)

        print("Time to Train....")

        print (np.asarray(training))

        X = np.array(training)
        y = np.array(output)
        start_time = time.time()
        # this line will call the train method and produce the synapses json file.
        train(X, y, 25, 0.1, 100000, False, 0.2)

        elapsed_time = time.time() - start_time
        print("processing time:", elapsed_time, "seconds")
        print("Got some Gainz")

        # probability threshold
        ERROR_THRESHOLD = 0.2
        # load our calculated synapse values
        synapse_file = 'synapses.json'

        with open(synapse_file) as data_file:
            synapse = json.load(data_file)
            synapse_0 = np.asarray(synapse['synapse0'])
            synapse_1 = np.asarray(synapse['synapse1'])

        def classify(sentence, show_details=False):
            results = think(sentence, show_details)
            results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
            results.sort(key=lambda x: x[1], reverse=True)
            return_results = [[classes[r[0]], r[1]] for r in results]
            print("%s \n classification: %s" % (sentence, return_results))
            return return_results


        classify("Who is professor Goel")

        classify("Who is Ben")

        classify("What is the limit of words on an assignment")

        classify("How many projects are in this class")


        garbage = 1234
        # for each question, we want to build a "Frame" which represents what I am asking.
        # each frame will then relate to possible answers. As the user provides feedback,
        # new frames can be created or existing frames can be changed to solidify the answer.


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

