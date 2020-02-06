#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from csv import reader
with open("C:/Users/iiitb/Desktop/text_Emotion.csv", 'r') as f:
    data = list(reader(f)) #Imports the CSV
    

print(len(data))
print(data[10])


# In[ ]:


texts = []
for i, point in enumerate(data):
    texts.append(data[i][3])
texts.pop(0)
print(texts[0])


# In[ ]:


labels = []
for i, point in enumerate(data):
    labels.append(data[i][1])
labels.pop(0)
print(labels[0])


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

text_train, text_test, label_train, label_test = train_test_split(texts, labels,test_size=0.2)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, max_iter=5, random_state=42)),
                     ])
fitting = text_clf.fit(text_train, label_train)
res = text_clf.predict(text_test)
print(res)


# In[ ]:


# turn the set in an array of boolean T|F and take the average of those
# 1.0 = perfect score, 0.0 is all mismatched
# print(len(res))
# print(len(label_test))
mn = np.mean(res == label_test)
print(mn)


# In[ ]:


predicted_svm = text_clf.predict(text_test)
np.mean(predicted_svm == label_test)


# In[ ]:


with open("C:/Users/iiitb/Downloads/lotr.txt","r")as sw:
    test_data=sw.read().split("\n")
for text in test_data:
    print(text)
    print(text_clf.predict([text]))


# In[ ]:


predicted_svm = text_clf.predict(test_data)
predicted_svm


# In[ ]:


predicted_svm = list(predicted_svm)
predicted_svm


# In[ ]:


feature_cont_input = [10027,15260,8437,2612,16041,2938,6116,210,1311,338,1501,2529,3869]
feature_cont_predicted = [predicted_svm.count('sadness'),predicted_svm.count('neutral'), predicted_svm.count('happiness'),predicted_svm.count('relief'),predicted_svm.count('worry'),predicted_svm.count('fun'),predicted_svm.count('love'),predicted_svm.count('anger'),predicted_svm.count('enthusiasm'),predicted_svm.count('boredom'),predicted_svm.count('empty'),predicted_svm.count('hate'),predicted_svm.count('surprise')]


# In[ ]:


print(feature_cont_input)
print(feature_cont_predicted)


# In[ ]:


np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()


# Example data
people = ('Sadness', 'Neutral', 'Happiness', 'Relief', 'Worry','Fun','Love','Anger','Enthusiasm','Boredom','Empty','Hate','Surprise')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, feature_cont_predicted, xerr=error, align='center')
ax.set_yticks(y_pos)

ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Count of Labels')
ax.set_title('Count of predicted emotions')

plt.show()


# In[ ]:


labels = ['Sadness', 'Neutral', 'Happiness', 'Relief', 'Worry','Fun','Love','Anger','Enthusiasm','Boredom','Empty','Hate','Surprise']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2,feature_cont_input, width, label='Actual Value')
rects2 = ax.bar(x + width/2, feature_cont_predicted, width, label='Predicted Value')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Emotions')
ax.set_title('Emotion mining')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# In[ ]:





# In[ ]:




