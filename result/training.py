import seaborn as sn 
import pandas as pa 
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn. metrics import classification_report
y_true = []
y_pred = []
labels = ["Bacteria", "Covid-19", "Lung Opacity",
"Normal"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
print (cm)
df_cm = pd. DataFrame (cm, labels, labels)
ax = sn. heatmap (df_cm, annot=True, annot_kws={"size": 16}, square=True, bar=False, fmt='g')
ax.set_ylim(0, 4) 
plt.xlabel("Predicted") 
plt.ylabel("Actual") 
ax.invert_yaxis ()
# optional
# show the precision and recall score
print(classification_report(y_true, y_pred, digits=4))
plt.show()