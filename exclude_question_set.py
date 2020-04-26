
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#from nltk import word_tokenize
#from nltk.stem import WordNetLemmatizer
import numpy as np
#from nltk.stem import WordNetLemmatizer
import math
from statistics import mean, stdev
import re
import os

remove = True

def exclude_questions_from_set(examples, set_path):

    with open(set_path) as remove_file:
        remove_questions = [reduce_question(q) for q in remove_file.readlines()]

    remove_count = 0

    filtered_examples = []
    for conversation in examples:
        filtered_conversation = []
        for turn in (conversation):
            if all([r != reduce_question(turn[0]) for r in remove_questions]):
                filtered_conversation.append(turn)
            else:
                remove_count += 1
    
        filtered_examples.append(filtered_conversation)
    
    print('Average removed {}'.format(remove_count / float(len(examples))))

    return filtered_examples


def reduce_question(question):
    question_prefixes = get_question_prefixes()
    base = re.sub(' +', ' ', question.strip())
    for word in base.split(' '):
        if ('_' in word and 'l_a' not in word) or any([char.isdigit() for char in word]):
            base = base.replace(word, '')

    for _ in range(2):
        for prefix in question_prefixes:
            if base[:len(prefix)] == prefix and len(base) > len(prefix):
                base = base.replace(prefix, '').strip()

    return base.strip()

def get_question_prefixes():
    # These are randomly inserted at the beginning of questions, if I do not remove them the number of unique questions
    # become intractable
    question_prefixes = [
        'mhm',
        '<research assistant speaking>'
        'hmm',
        'laughter',
        '<laughter>',
        '[laughter]',
        'yeah',
        'yes',
        'i see what you mean',
        'i see',
        'mm',
        'nice',
        'oh no',
        'oh',
        'uh oh',
        'oh my gosh',
        'okay',
        'right',
        'wow',
        'uh huh',
        'uh huh uh huh',
        'that\'s so good to hear',
        'that\'s great',
        'that\'s good',
        'that makes sense',
        'that sucks',
        'that sounds really hard',
        'that sounds like a great situation',
        'that sounds interesting',
        'right',
        'really',
        'i\'m sorry to hear that',
        'i\'m sorry',
        'i understand',
        'cool',
        'aww',
        'awesome',
        'aw',
    ]
    return question_prefixes
