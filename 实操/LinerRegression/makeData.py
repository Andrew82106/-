import random

numOfX = 7
numOfInstance = int(1e4)


def function(X_List):
    weight = [2**i for i in range(numOfX)]
    assert len(weight) == len(X_List)
    y = sum([weight[index]*X_List[index] for index in range(len(X_List))])
    return y


def generateData():
    X = []
    Y = []
    for i in range(numOfInstance):
        X.append([])
        for j in range(numOfX):
            X[-1].append(random.randint(1, 100000))
        Y.append(function(X[-1]))
    return X, Y



if __name__ == '__main__':
    X, Y = generateData()
    print("end")