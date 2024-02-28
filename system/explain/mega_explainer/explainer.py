"""Compares many explanations to determine the best one."""
import copy
from dataclasses import dataclass
from functools import partial
from typing import Union, Any

import heapq
import numpy as np
import pandas as pd
import torch

from explain.mega_explainer.shap_explainer import SHAPExplainer


@dataclass
class MegaExplanation:
    """The return format for the mega explanation!"""
    list_exp: list
    score: float
    label: int
    best_explanation_type: str
    agree: bool


def conv_disc_inds_to_char_enc(discrete_feature_indices: list[int], n_features: int):
    """Converts an array of discrete feature indices to a char encoding.

    Here, the ith value in the returned array is 'c' or 'd' for whether the feature is
    continuous or discrete respectively.

    Args:
        discrete_feature_indices: An array like [0, 1, 2] where the ith value corresponds to
                                  whether the arr[i] column in the data is discrete.
        n_features: The number of features in the data.
    Returns:
        char_encoding: An encoding like ['c', 'd', 'c'] where the ith value indicates whether
                       that respective column in the data is continuous ('c') or discrete ('d')
    """
    # Check to make sure (1) feature indices are integers and (2) they are unique
    error_message = "Features all must be type int but are not"
    assert all(isinstance(f, int) for f in discrete_feature_indices), error_message
    error_message = "Features indices must be unique but there are repetitions"
    assert len(set(discrete_feature_indices)) == len(discrete_feature_indices), error_message
    # Perform conversion
    char_encoding = ['e'] * n_features
    for i in range(len(char_encoding)):
        if i in discrete_feature_indices:
            char_encoding[i] = 'd'
        else:
            char_encoding[i] = 'c'
    # In case something still went wrong
    assert 'e' not in char_encoding, 'Error in char encoding processing!'
    return char_encoding


class Explainer:
    """
    Explainer is the orchestrator class that drives the logic for selecting
    the best possible explanation from the set of explanation methods.
    """

    def __init__(self,
                 explanation_dataset: np.ndarray,
                 explanation_model: Any,
                 feature_names: list[str],
                 discrete_features: list[int],
                 use_selection: bool = True):
        """
        Init.

        Args:
            explanation_dataset: background data, given as numpy array
            explanation_model: the callable black box model. the model should be callable via
                               explanation_model(data) to generate prediction probabilities
            feature_names: the feature names
            discrete_features: The indices of the discrete features in the dataset. Note, in the
                               rest of the repo, we adopt the terminology 'categorical features'.
                               However, in this mega_explainer sub folder, we adopt the term
                               `discrete features` to describe these features.
            use_selection: Whether to use the explanation selection. If false, uses lime.
        """
        if isinstance(explanation_dataset, pd.DataFrame):
            # Creating a copy of the explanation dataset... For large datasets, this may be an
            # issue. However, converting from pd.DataFrame to np.ndarray in this way seems
            # to overwrite the underlying dataset, causing potentially confusing issues
            explanation_dataset = copy.deepcopy(explanation_dataset)
            explanation_dataset = explanation_dataset.to_numpy()
        else:
            arr_type = type(explanation_dataset)
            message = f"Data must be pd.DataFrame or np.ndarray, not {arr_type}"
            assert isinstance(explanation_dataset, np.ndarray), message

        self.data = explanation_dataset
        self.model = explanation_model
        self.feature_names = feature_names

        # We store a dictionary containing all the explanation methods we are going to compare
        # in order to figure out "the best" explanation. These methods are initialized and
        # stored here

        available_explanations = {}

        # add shap
        shap_explainer = SHAPExplainer(self.model, self.data)
        available_explanations["shap"] = shap_explainer

        self.explanation_methods = available_explanations

        # This is a bit clearer, instead of making users use this representation + is the way
        # existing explanation packages (e.g., LIME do it.)
        self.feature_types = conv_disc_inds_to_char_enc(discrete_feature_indices=discrete_features,
                                                        n_features=self.data.shape[1])


    @staticmethod
    def _arr(x) -> np.ndarray:
        """Converts x to a numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    @staticmethod
    def check_exp_data_shape(data_x: np.ndarray) -> np.ndarray:
        """Checks to make sure the data being explained is a single instance and 1-dim."""
        # Check to make sure data_x is an individual sample
        data_x_shape = data_x.shape
        if len(data_x_shape) > 1:
            n_samples = data_x_shape[0]
            if n_samples > 1:
                message = f"Data must be individual sample, but has shape {data_x_shape}"
                assert len(data_x_shape) == 1, message
        elif len(data_x_shape) == 1:
            data_x = data_x.reshape(1, -1)
        return data_x

    def explain_instance(self,
                         data: Union[np.ndarray, pd.DataFrame]) -> MegaExplanation:
        """Computes the explanation.

        This function computes the explanation. It calls several explanation methods, computes
        metrics over the different methods, computes an aggregate score and returns the best one.

        Args:
            top_k_ending_pct:
            top_k_starting_pct:
            data: The instance to explain. If given as a pd.DataFrame, will be converted to a
                  np.ndarray
        Returns:
            explanations: the final explanations, selected based on most faithful
        """
        if not isinstance(data, np.ndarray):
            try:
                data = data.to_numpy()
            except Exception as exp:
                message = f"Data not type np.ndarray, failed to convert with error {exp}"
                raise NameError(message)

        explanations, scores = {}, {}

        # Makes sure data is formatted correctly
        formatted_data = self.check_exp_data_shape(data)

        # Explain the most likely class
        label = self.model(formatted_data)[0]

        # Iterate over each explanation method and compute fidelity scores of topk
        # and non-topk features per the method
        for method in self.explanation_methods.keys():
            cur_explainer = self.explanation_methods[method]
            cur_expl, score = cur_explainer.get_explanation(formatted_data)

            explanations[method] = cur_expl.squeeze(0)
            scores[method] = score

        best_method = "shap"
        best_exp = explanations[best_method]
        best_method_score = scores[best_method]
        agree = True

        # Format return
        # TODO(satya,dylan): figure out a way to get a score metric using fidelity
        final_explanation = self._format_explanation(best_exp.numpy(),
                                                     label,
                                                     best_method_score,
                                                     best_method,
                                                     agree)

        return final_explanation

    def _format_explanation(self, explanation: list, label: int, score: float, best_method: str, agree: bool):
        """Formats the explanation in LIME format to be returned."""
        list_exp = []

        # combine feature importances & features names into tuples of feature name and feature
        # importance
        for feature_name, feature_imp in zip(self.feature_names, explanation):
            list_exp.append((feature_name, feature_imp))

        # Sort the explanations so that the most important features are first
        list_exp.sort(key=lambda x: abs(x[1]), reverse=True)

        # Format the output
        return_exp = MegaExplanation(list_exp=list_exp,
                                     label=label,
                                     score=score,
                                     best_explanation_type=best_method,
                                     agree=agree)

        return return_exp
