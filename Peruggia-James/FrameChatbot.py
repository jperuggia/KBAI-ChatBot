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
from textblob import TextBlob
import operator



def generate_question_answer(s):
    question = s.split("?")[0]
    answer = s.split("?")[1]
    fa = {
        "question": question,
        "answer": answer
    }
    return fa


class Chatbot:

    def __init__(self,FAQPathFilename):
        # FAQPathFilename is string containing
        # path and filename to text corpus in FAQ format.
        self.FAQPathFilename = FAQPathFilename
        with open(FAQPathFilename,"r", encoding="utf-8") as f: # Example code
            self.FAQasList = f.readlines()                     # Example code

        # time to let the network learn.
        self.KnowledgeBase = self.learn()

        return

    def learn(self):
        knowledge_base = {}
        i = 0
        for s in self.FAQasList:
            base_info = generate_question_answer(s)
            base_info = self.generate_knowledge_item(base_info)
            knowledge_base[i] = base_info
            i += 1

        return knowledge_base

    # takes a string from the FAQ and returns the question and answer in a dictionary.

    def check_for_previous_frame(self):
        return False

    def add_possible_answer_to_frame(self, answer):
        return True

    def generate_knowledge_item(self, base_info, check_base=True):
        if self.check_for_previous_frame() and check_base:
            # the frame with similar data points exists. Something must be done!
            return False

        # Time to create a new knowledge item. A question the Agent has never seen before!
        question_tb = TextBlob(base_info["question"])
        blob_tags = question_tb.tags

        if check_base:
            answer_str = base_info["answer"]
        else:
            answer_str = ''

        frame = {
            'nouns': [],
            'pronouns': [],
            'verb': [],
            'det': [],
            'digits': [], #cardinal digits
            #
            "possibleAnswer": [],
            "otherInfo": [],
            "inputQuestion": base_info["question"].lower(),
            "answer": answer_str,
            "wh_det": {
                "who": [],
                "what": [],
                "when": [],
                "how": [],
                "which": [],
                "where": [],
                "why": []
            },
            # possible extension
            "parent_frames": [],
            "child_frames": []
        }

        for t in blob_tags:
            word = ''.join(t[0])
            tag = t[1]

            if tag.startswith('V'):
                frame["verb"].append(word.lower())
            elif tag.startswith('N'):
                frame["nouns"].append(word.lower())
            elif tag.startswith('PR'):
                frame["pronouns"].append(word.lower())
            elif tag.startswith('CD'):
                frame["digits"].append(word.lower())
            elif tag.startswith('DT'):
                frame["det"].append(word.lower())
            elif tag.startswith('WD') or tag.startswith('WR') or tag.startswith('WP'):
                wd = word.lower()
                frame["wh_det"][wd].append(answer_str)
            else:
                frame["otherInfo"].append(word.lower())
        return frame

    def find_most_similar_frame(self, question_frame):
        knowledge_base_size = len(self.KnowledgeBase)
        i = 0
        percent_likely = {}
        while i < knowledge_base_size:
            percent_likely[i] = self.question_fitness(i, question_frame)
            if percent_likely[i] == float("inf"):
                break #stop execution
            i += 1

        return percent_likely

    def question_fitness(self,knowledge_base_index, question_frame):
        # define the fitness based on the frames "knowledge"
        total_points = 0
        knowledge_frame = self.KnowledgeBase[knowledge_base_index]

        if question_frame["inputQuestion"] == knowledge_frame["inputQuestion"]:
            return float("inf") #this is the answer

        # find total points for frame
        for n in question_frame["nouns"]:
            if n in knowledge_frame["nouns"]:
                total_points += 5

        for n in question_frame["pronouns"]:
            if n in knowledge_frame["pronouns"]:
                total_points += 5

        for n in question_frame["verb"]:
            if n in knowledge_frame["verb"]:
                total_points += 5

        for n in question_frame["det"]:
            if n in knowledge_frame["det"]:
                total_points += 5

        for n in question_frame["digits"]:
            if n in knowledge_frame["digits"]:
                total_points += 3

        for n in question_frame["otherInfo"]:
            if n in knowledge_frame["otherInfo"]:
                total_points += 1


        for n in question_frame["wh_det"]:
            # if the answer is found, we want to use it. If multiple, we need to go down more frames.
            if len(knowledge_frame["wh_det"][n]) > 0:
                total_points + 10

        return total_points

    def ask_question(self, question):
        q_a_frame = {
            "question": question
        }
        search_frame = self.generate_knowledge_item(q_a_frame, False)
        results = self.find_most_similar_frame(search_frame)

        v = list(results.values())
        rt = list(results.keys())[v.index(max(v))]

        return self.KnowledgeBase[rt]["answer"]

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
        # for qa in self.FAQasList:           # Example code
        #     question = qa.split('?')[0]     # Example code
        #     answer =qa.split('?')[1]        # Example code
        #     if question == msg:
        #         response = answer
        #         break

        if response == '':
            response = self.ask_question(msg)

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