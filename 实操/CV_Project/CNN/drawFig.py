import matplotlib.pyplot as plt


def draw_line_chart(D, X='test_x', Y='test_y', title='test_title', saveLocation=None):
    plt.clf()
    plt.plot(D, marker='o')
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.title(title)
    if saveLocation is not None:
        plt.savefig(saveLocation)
    else:
        plt.show()


if __name__ == '__main__':
    data = [1, 2, 3, 4, 5]
    draw_line_chart(data)
