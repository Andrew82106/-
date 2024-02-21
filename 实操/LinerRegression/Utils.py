import matplotlib.pyplot as plt


def plot_paraLst(paraLst):
    plt.figure(figsize=(10, 6))
    for i in range(len(paraLst)):
        plt.plot(paraLst[i], label=f'para {i}')
    plt.xlabel('Time')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Value vs TrainingTimes')
    plt.legend()
    plt.show()


def plot_muti_Line(paraDict, X_Label='Time', Y_Label='Value', title='Value vs TrainingTimes'):
    plt.figure(figsize=(10, 6))
    for name in paraDict:
        plt.plot(paraDict[name], label=f'para {name}')
    plt.xlabel(X_Label)
    plt.ylabel(Y_Label)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_paraLst([
        [1, 2, 3],
        [1, 1, 3],
        [4, 6, 7]
    ])

    plot_muti_Line({
        'name1': [1, 2, 3],
        'name2': [1, 1, 3],
        'name3': [4, 6, 7]
    })
