import random

numOfX = 1
numOfInstance = int(1e4)


def function(X_List):
    weight = [2**i for i in range(numOfX)]
    assert len(weight) == len(X_List)
    y = sum([weight[index]*X_List[index] for index in range(len(X_List))])
    return y


def generateData():
    X = []
    Y = []
    sumX = 0
    for i in range(numOfInstance):
        X.append([])
        for j in range(numOfX):
            X[-1].append(random.randint(1, 100000))
            sumX += X[-1][-1]
        Y.append(function(X[-1]))
    print(f"x range:[{min(X)}, {max(X)}],average x:{sumX/len(X)}, y range:[{min(Y)}, {max(Y)}]")
    return X, Y



if __name__ == '__main__':
    X, Y = generateData()
    print("end")