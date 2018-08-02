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
import numpy
import re


stemmer = nltk.SnowballStemmer("english")
lamma = WordNetLemmatizer()


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_output_to_derivative(output):
    return output*(1-output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence.lower())
    # stem each word
    sentence_words = [lamma.lemmatize(w) for w in sentence_words]
    sentence_words = [stemmer.stem(w) for w in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return numpy.array(bag)


# Method to determine if the sentence contains
# an Assignment # or Project # word. To help with accuracy
# the bot will combine these to be a single word.
def detect_project_or_assignment(sentence):
    sentence = sentence.lower()
    a = re.search(r'\b(project)\b\s\d+', sentence)
    p = re.search(r'\b(assignment)\b\s\d+', sentence)

    if a is not None and a.start() >= 0:
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

    def __init__(self,faq_path_filename):
        self.faq_path_filename = faq_path_filename
        self.faq_as_list = []
        # FAQPathFilename is string containing
        # path and filename to text corpus in FAQ format.
        self.ERROR_THRESHOLD = 0.10

        # our learned data set. The black box of NO!
        self.QUESTION_ASKED = ""
        self.ANSWER_RETURNED = ""
        self.INCORRECT_ANSWERS = {}

        # The neural network will use these to classify inputs and outputs.
        self.documents = []
        self.classes = []
        self.words = []
        self.ignore_words = ["?", ".", "!", ",", "'s", "2"]
        self.s_words = set(stopwords.words('english'))

        # the training data for the NN
        self.training = []
        self.output = []

        # dictionary to hold EXACT question and answer matches for quick retrieval of exact matches.
        self.exact_q_and_a = {}

        # network layers.
        self.synapse_0 = []
        self.synapse_1 = []

        # network variables
        self.hidden_neurons = 15
        self.alpha = 0.1
        self.iterations = 30000

        self.parse_corpus()  # always parse the corpus.
        self.let_the_learning_begin()

    def let_the_learning_begin(self):
        self.learn_from_corpus()
        self.create_training_data()
        x = numpy.array(self.training)
        y = numpy.array(self.output)
        self.train_network(x, y)

    def parse_corpus(self):

        with open(self.faq_path_filename, "r", encoding="utf-8") as f:  # Example code
            self.faq_as_list = f.readlines()  # Example code

        self.faq_as_list[len(self.faq_as_list)-1] += '\n'

    def learn_from_corpus(self):
        # clear some stuff out
        self.words = []
        self.documents = []
        self.classes = []
        training_set = []

        for s in self.faq_as_list:
            question = s.split("?")[0].lower()
            answer = s.split("?")[1].rstrip()
            training_set.append({"answer": answer, "question": question})
            self.exact_q_and_a[question] = answer
            # if question not in self.exact_q_and_a:


        # loop over each sentence in training data.
        for pattern in training_set:
            pattern["question"] = detect_project_or_assignment(pattern["question"])
            q_words = nltk.word_tokenize(pattern["question"])
            # q_words = [w for w in q_words if w not in self.s_words]
            q_words = [lamma.lemmatize(w) for w in q_words]
            q_words = [stemmer.stem(w) for w in q_words]
            q_words = [w.lower() for w in q_words if w not in self.ignore_words]
            self.words.extend(q_words)
            self.documents.append((q_words, pattern["answer"]))
            # add class to list if not already there.
            if pattern["answer"] not in self.classes:
                self.classes.append(pattern["answer"])

        self.words = list(set(self.words))
        # sort it!
        self.words = sorted(self.words)
        self.classes = list(set(self.classes))

    def create_training_data(self):
        self.training = []
        self.output = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]

            # create our bag of words array
            for w in self.words:
                if w in pattern_words:
                    bag.append(1)
                else:
                    bag.append(0)

            self.training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            self.output.append(output_row)

    def train_network(self, X, y):

        self.hidden_neurons = int((len(X) + len(y))/2)
        # self.hidden_neurons = round(int(len(self.words)/3),1)
        print("The chatbot is learning.. please be patient")
        numpy.random.seed(1)
        last_mean_error = 1
        s0 = 2 * numpy.random.random((len(X[0]), self.hidden_neurons)) - 1
        s1 = 2 * numpy.random.random((self.hidden_neurons, len(self.classes))) - 1

        prev_synapse_0_weight_update = numpy.zeros_like(s0)
        prev_synapse_1_weight_update = numpy.zeros_like(s1)

        synapse_0_direction_count = numpy.zeros_like(s0)
        synapse_1_direction_count = numpy.zeros_like(s1)

        for j in iter(range(self.iterations + 1)):
            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(numpy.dot(layer_0, s0))
            layer_2 = sigmoid(numpy.dot(layer_1, s1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j % 10000) == 0 and j > 5000:
                # print("Itter " + str(j))
                # if the error is getting worse stop.
                if numpy.mean(numpy.abs(layer_2_error)) < last_mean_error:
                    last_mean_error = numpy.mean(numpy.abs(layer_2_error))

            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(s1.T)
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if j > 0:
                synapse_0_direction_count += numpy.abs(
                    ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += numpy.abs(
                    ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

            s1 += self.alpha * synapse_1_weight_update
            s0 += self.alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        # set the global self to the things that are needed for synapse.
        self.synapse_1 = numpy.asarray(s1)
        self.synapse_0 = numpy.asarray(s0)

    def think(self, sentence):
        # clean up the question into usable data.
        sentence = detect_project_or_assignment(sentence)
        x = bag_of_words(sentence, self.words)

        level_0 = x
        level_1 = sigmoid(numpy.dot(level_0, self.synapse_0))
        level_2 = sigmoid(numpy.dot(level_1, self.synapse_1))

        return level_2

    def classify(self, sentence):
        results = self.think(sentence)
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results = [[self.classes[r[0]], r[1]] for r in results]
        return return_results

    def user_feedback(self, yesorno, updated_response):

        # if the answer was correct, lets add this to the corpus for future training sets.
        if yesorno and len(self.ANSWER_RETURNED) > 0 and self.ANSWER_RETURNED != "I do not know.":
            # the correct answer was given, lets add this to our short term knowledge.
            self.exact_q_and_a[self.QUESTION_ASKED] = self.ANSWER_RETURNED
            # add this to the corpus for future training!
            self.add_to_faq(self.ANSWER_RETURNED)

        if len(updated_response) > 0:
            # print("Time to add a new response to the question and retrain the network")
            self.improve_knowledge_base(updated_response)
        if not yesorno:
            # print("Wrong answer, didn't give correct one, how do you retrain?")
            self.incorrect_answer_fix()

        return

    def incorrect_answer_fix(self):
        if self.QUESTION_ASKED not in self.INCORRECT_ANSWERS:
            self.INCORRECT_ANSWERS[self.QUESTION_ASKED] = []

        self.INCORRECT_ANSWERS[self.QUESTION_ASKED].append(self.ANSWER_RETURNED)

    def add_to_faq(self, answer):
        updated_question = detect_project_or_assignment(self.QUESTION_ASKED.lower())
        if not updated_question.endswith('?'):
            updated_question += "?"

        self.faq_as_list.append(updated_question + answer + "\n")

    def improve_knowledge_base(self, expected_answer):
        self.add_to_faq(expected_answer)
        self.let_the_learning_begin()
        # loop retrain if the question still doesn't produce the correct output?

    def check_short_term_memory(self, msg):
        # due to some strange thigns with keys, check for "partial" substrings that might
        # be found, in case of half questions or extra ? at the end.
        use_key = msg
        for k in self.exact_q_and_a:
            if msg in k:
                use_key = k

        if use_key in self.exact_q_and_a:
            return self.exact_q_and_a[use_key]
        else:
            return None

    def input_output(self, msg):
        # save the question off in the memory for learning based on feedback.
        self.QUESTION_ASKED = msg.lower()
        if self.QUESTION_ASKED.endswith("?"):
            self.QUESTION_ASKED = self.QUESTION_ASKED[:-1]

        response = self.check_short_term_memory(self.QUESTION_ASKED)
        if response is None:
            r = self.classify(msg)
            # remove from R the answers if they are in the black box for the question.
            if len(r) > 0:
                if self.QUESTION_ASKED in self.INCORRECT_ANSWERS:
                    known_incorrect_for_question = self.INCORRECT_ANSWERS[self.QUESTION_ASKED]
                    for possible_answer in r:
                        if possible_answer[0] in known_incorrect_for_question:
                            r.remove(possible_answer)

            # get the best value from R that I can find.
            response = "I do not know."
            if len(r) > 0:
                response = max(r, key=lambda item: item[1])[0]

        self.ANSWER_RETURNED = response

        return response + '\n'