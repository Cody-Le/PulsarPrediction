import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def featureScale(x):

    average = np.mean(x, axis=0)
    range = np.max(x, axis = 0) - np.min(x, axis = 0)
    return (x-average)/range


class logisticRegression():
    def __init__(self, inputSize):
        self.theta = np.zeros((inputSize + 1,1))
        self.inputShape = inputSize + 1

        #print(self.theta)



    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def predict(self, x):
        X = np.ones((x.shape[0], x.shape[1] + 1))

        X[:,1:] = x

        linearHypothesis = np.dot(X,self.theta)
        return self.sigmoid(linearHypothesis)

    def cost(self, y_pred, y):
        totalCost = -1 * np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return totalCost



    def training(self, x, y, epochs, learningRate):
        X = np.ones((x.shape[0], x.shape[1] + 1))

        X[:, 1:] = x
        costOverTime = np.zeros(epochs)
        for epoch in range(epochs):
            newTheta = np.zeros(self.theta.shape)
            for j in range(self.inputShape):
                gradient = np.mean((self.predict(x) - y)* X[:,j][:, np.newaxis])

                newTheta[j] = self.theta[j] -  learningRate * gradient
            self.theta = newTheta

            print(epoch, self.cost(self.predict(x), y))
            costOverTime[epoch] = self.cost(self.predict(x), y)
        plt.figure()
        plt.plot(np.arange(0, epoch + 1), costOverTime)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.show()





#Processing Data

df = pd.read_csv("./pulsar_data_train.csv")
df_np = df.to_numpy()
X = df.iloc[:,:-1].to_numpy()
#Clean data
X[np.isnan(X)] = 0

X_scaled = featureScale(X)


#Split data into training set and evalulation set

X_train = X_scaled[627:]
y_train = df.iloc[627:,-1].to_numpy()

X_eval = X_scaled[0:627]
y_eval= df.iloc[0:627,-1].to_numpy()

#Set up model

model = logisticRegression(8)

#print(model.cost(y_pred, y_train[:, np.newaxis]))
model.training(X_train, y_train[:, np.newaxis], 10000, 0.01)
y_pred = model.predict(X_eval)
cost = model.cost(y_pred, y_eval)
print("The Final cost",cost)