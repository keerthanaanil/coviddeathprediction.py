import numpy as np
import pandas as pd


class RecoveryPrediction(object):

    def __init__(self, k):
        self.k = k

    @staticmethod
    # DEFINING FUNCTION TO FIND THE DISTANCE BETWEEN TWO VECTORS
    def dist(aa, b):
        return np.sqrt(sum((aa - b) ** 2))

    # DEFINING CONDITION
    def neighbour(self, x, y, q, k=3):
        vals = []
        m = x.shape[0]  # Here m is 100 representing no. of entries

        for A in range(m):
            d = self.dist(q, x[A])
            vals.append((d, y[A]))
        vals = sorted(vals)

        # DETERMINING THE FIRST K POINTS
        vals = vals[:k]
        vals = np.array(vals)

        # DETERMINING THE UNIQUE VALUES
        new_vals = np.unique(vals[:, 1], return_counts=True)  # Numpy arrays of unique values and their counts
        index = new_vals[1].argmax()
        pred = new_vals[0][index]
        return pred

    @staticmethod
    # DEFINING FUNCTION TO TEST THE ACCURACY FOR A GIVEN 'k' VALUE
    def accuracy(y_check, y_pred):
        n_correct = 0
        for p, r in zip(y_check, y_pred):
            if p == r:
                n_correct += 1
        return n_correct / len(y_check)

    @staticmethod
    # DEFINING FUNCTION TO SPLIT A GIVEN DATA FOR TRAINING AND TESTING
    def split(dataset, test_size=0.25):
        num = int(len(dataset) * test_size)
        test_data = dataset.sample(num)
        train_data = []
        for ind in dataset.index:
            if ind in test_data.index:
                continue
            train_data.append(dataset.iloc[ind])
        train_data = pd.DataFrame(train_data)
        return train_data, test_data


# CONVERTING TO DATAFRAMES
dfx = pd.read_csv(r'/home/hp/Downloads/train.csv')
dftest = pd.read_csv(r'/home/hp/Downloads/test.csv')

knn = RecoveryPrediction(k=3)

# SPLITTING THE TRAINING SET FOR CHECKING THE ALGORITHM
train_set, test_set = knn.split(dfx)
test_x = test_set.drop(['NAME', 'RECOVERY'], axis=1)
test_y = test_set.drop(dfx.loc[:, 'NAME':'KIDNEY'].columns, axis=1)
train_x = train_set.drop(['NAME', 'RECOVERY'], axis=1)
train_y = train_set.drop(dfx.loc[:, 'NAME':'KIDNEY'].columns, axis=1)

check = []
train_x = train_x.values
train_y = train_y.values
test_x = test_x.values
for i in range(len(test_x)):
    a = int(knn.neighbour(train_x, train_y, test_x[i]))
    check.append(a)
check = np.array(check)
testy = test_y.values
k_value = knn.accuracy(check, testy)
print("\n")
print(k_value)
print("This is the accuracy for suggested k = 3.")
print("\n")

# REMOVING THE COLUMNS NOT NECESSARY FOR CALCULATIONS
x = dfx.drop(['NAME', 'RECOVERY'], axis=1)
y = dfx.drop(dfx.loc[:, 'NAME':'KIDNEY'].columns, axis=1)

# GENERATING NUMPY ARRAYS
x = x.values
y = y.values
y = y.reshape((-1,))

print(dfx)

# PREDICTING THE RECOVERY FOR THE TESTING SET
result = []
q = dftest.drop(['NAME'], axis=1)
q = q.values
q = q[:, 0:]
n = q.shape[0]
for j in range(n):
    a = int(knn.neighbour(x, y, q[j]))
    result.append(a)
result = np.array(result)

# PRINTING THE OUTPUT
print(dftest)
print(result)
dftest['PREDICTED_RECOVERY'] = result
print(dftest)
