import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


def plot_outliers(data, predictions):
    ax = plt.axes(projection='3d')
    #ax = plt.axes()

    predictions = np.array(predictions)
    colors = np.array(["red", "green"])
    #ax.scatter(data[:, 0], data[:, 1], c=colors[predictions])
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=colors[predictions])

    plt.show()

    """
    #
    # zline = np.linspace(0, 15, 1000)
    # xline = np.sin(zline)
    # yline = np.cos(zline)
    # ax.plot3D(xline, yline, zline, 'gray')
    non_anomalies = []
    anomalies = []
    for row, label in zip(data, predictions):
        if label == 1:
            anomalies.append(row)
        else:
            non_anomalies.append(row)



    non_anomalies=np.array(non_anomalies)
    anomalies = np.array(anomalies)

    # distance_not_anomaly, src_not_anomaly, dest_not_anomaly = non_anomalies[:, 0],non_anomalies[:, 1],non_anomalies[:, 2]
    # distance_anomaly, src_anomaly, dest_anomaly = anomalies[:, 0], anomalies[:, 1], anomalies[:,2]

    non_anomaly_columns = non_anomalies[:, 0], non_anomalies[:, 1], non_anomalies[:, 2]
    anomaly_columns = anomalies[:, 0], anomalies[:, 1], anomalies[:, 2]

    #print(predictions)
    #print(len(anomalies))

    #predictions=list(map(lambda x: 1 if x == 1 else -1, predictions))

    #cmap = matplotlib.colors.ListedColormap(['red', 'green'])
    predictions=np.array(predictions)
    
    # ax.scatter(*non_anomaly_columns,color='g')
    # ax.scatter(*anomaly_columns, color='r')

    plt.show()
    """


features = ["record ID", "duration_", "src_bytes", "dst_bytes"]

df = pd.read_csv('conn_attack.csv', names=features, header=None)
df_labels = pd.read_csv('conn_attack_anomaly_labels.csv', names=["record ID", "malicious?"], header=None)

dataset = df.to_numpy()
dataset = np.delete(dataset, 0, 1)

labels = df_labels.to_numpy()
labels = np.delete(labels,0,1).flatten()

duration_train, duration_test, src_train, src_test, dest_train, dest_test, labels_train, labels_test = \
    train_test_split(dataset[:, 0], dataset[:, 1], dataset[:, 2],labels , test_size=0.2,shuffle=True)

x_train = np.stack((duration_train, src_train, dest_train), axis=1)
x_test = np.stack((duration_test, src_test, dest_test), axis=1)

iso_forest = IsolationForest(contamination=0.004,max_features=3,random_state=6,n_jobs=-1).fit(x_train)

y_pred1 = iso_forest.predict(x_test)
y_pred1=list(map(lambda x: 1 if x==-1 else 0,y_pred1))

print("accuracy1 score: {0:.2f}%".format(accuracy_score(labels_test, y_pred1) * 100))
print("precision1 score: {0:.2f}%".format(precision_score(labels_test, y_pred1) * 100))
print("recall1 score: {0:.2f}%".format(recall_score(labels_test, y_pred1) * 100))
print(confusion_matrix(labels_test, y_pred1))

plot_outliers(x_test,y_pred1)

#
#
# local_outlier_factor = LocalOutlierFactor(n_neighbors=20,contamination=0.004,n_jobs=-1).fit(x_train)
#
# y_pred2 = local_outlier_factor.fit_predict(x_test)
# y_pred2=list(map(lambda x: 1 if x==-1 else 0,y_pred2))
#
#
# print("accuracy2 score: {0:.2f}%".format(accuracy_score(labels_test, y_pred2) * 100))
# print("precision2 score: {0:.2f}%".format(precision_score(labels_test, y_pred2) * 100))
# print("recall2 score: {0:.2f}%".format(recall_score(labels_test, y_pred2) * 100))
# print(confusion_matrix(labels_test, y_pred2))
#
# plot_outliers(x_test,y_pred2)
#
#




