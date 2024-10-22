# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2024-10-14
Version: 1.0
"""

from transformers import pipeline
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def sentiment_analysis(list_of_headlines):
    labels = []  # To store sentiment labels
    scores = []  # To store sentiment scores

    for headline in list_of_headlines:
        # Perform sentiment analysis on each headline
        result = pipe(headline)

        # Assuming the result contains a list of dictionaries with 'label' and 'score'
        label = result[0]['label']  # Extract the 'label'
        score = result[0]['score']  # Extract the 'score'

        # Append the label and score to respective lists
        labels.append(label)
        scores.append(score)

    # Return the lists of labels and scores
    return labels, scores



#%% adding labels and scores to df
# df_sorted_clean['Label'] = labels
# df_sorted_clean['Score'] = scores

# class Person:
#     def __init__(self, name, age) -> object:
#         """
#
#         :rtype: object
#         :param name:
#         :param age:
#         """
#         self.name = name
#         self.age = age
#
#     def __str__(self):
#         """
#
#         :rtype: object
#         """
#         return f"{self.name}({self.age})"
