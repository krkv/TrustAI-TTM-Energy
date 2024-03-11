"""Show model mistakes"""
from copy import deepcopy

import gin
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from explain.actions.utils import get_parse_filter_text, get_rules

def average_difference(y_true, y_pred):
    """ Calculate average difference between y_true and y_pred"""
    difference_sum = np.sum(abs(y_true - y_pred))
    total_num = len(y_true)
    return round(difference_sum / total_num, 2)

def correct_pred(y_true, y_pred):
    return abs(y_true - y_pred) < 200
    

def one_mistake(y_true, y_pred, conversation, intro_text):
    """One mistake text"""
    label = y_true[0]
    prediction = y_pred[0]
        
    difference = round(abs(label - prediction), 2)

    return_string = (f"{intro_text} the model predicts <em>{str(prediction)}</em> and the ground"
                     f" label is <em>{str(label)}</em>, so the prediction is off by <b>{difference}</b>!")
    return return_string


def sample_mistakes(y_true, y_pred, conversation, intro_text, ids):
    """Sample mistakes sub-operation"""
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        difference_avg = average_difference(y_true, y_pred)
        total_num = len(y_true)
        return_string = (f"{intro_text} the model prediction is incorrect on average by {difference_avg} over {total_num} samples.")

    return return_string


def train_tree(data, target, depth: int = 1):
    """Trains a decision tree"""
    dt_string = []
    tries = 0
    while len(dt_string) < 3 and tries < 10:
        tries += 1
        dt = DecisionTreeClassifier(max_depth=depth).fit(data, target)
        dt_string = get_rules(dt,
                              feature_names=list(data.columns),
                              class_names=["correct", "incorrect"])
        depth += 1

    return dt_string


def typical_mistakes(data, y_true, y_pred, conversation, intro_text, ids):
    """Typical mistakes sub-operation"""
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        correct_vals = correct_pred(y_true, y_pred)
        incorrect_vals = correct_vals != True
        return_options = train_tree(data, incorrect_vals)

        if len(return_options) == 0:
            return "I couldn't find any patterns for mistakes the model typically makes."

        return_string = f"{intro_text} the model typically predicts incorrect:<br><br>"
        for rule in return_options:
            return_string += rule + "<br><br>"

    return return_string


@gin.configurable
def show_mistakes_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows the model mistakes."""
    data = conversation.temp_dataset.contents['X']
    y_true_pd = deepcopy(conversation.temp_dataset.contents['y'])

    if isinstance(y_true_pd, pd.Series):
        y_true = y_true_pd.to_numpy()
    elif isinstance(y_true_pd, list):
        y_true = np.array(y_true_pd)

    # Get ids
    ids = np.array(list(data.index))

    model = conversation.get_var('model').contents

    # The filtering text
    intro_text = get_parse_filter_text(conversation)

    if len(y_true) == 0:
        return "There are no instances in the data that meet this description.<br><br>", 0

    y_pred = model.predict(data)
    correct_vals = correct_pred(y_true, y_pred)
    if np.sum(correct_vals) == len(y_true):
        if len(y_true) == 1:
            return f"{intro_text} the model predicts correctly!<br><br>", 1
        else:
            return f"{intro_text} the model predicts correctly on all the instances in the data!<br><br>", 1

    if parse_text[i+1] == "sample":
        return_string = sample_mistakes(y_true,
                                        y_pred,
                                        conversation,
                                        intro_text,
                                        ids)
    elif parse_text[i+1] == "typical":
        return_string = typical_mistakes(data,
                                         y_true,
                                         y_pred,
                                         conversation,
                                         intro_text,
                                         ids)
    else:
        raise NotImplementedError(f"No mistake type {parse_text[i+1]}")

    return_string += "<br><br>"
    return return_string, 1
