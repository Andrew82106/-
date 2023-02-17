import os


def configLoc():
    return str(os.getcwd())[:str(os.getcwd()).find("动手深度学习")] + "动手深度学习/实操/基于softmax线性回归的性别识别系统/"


def configImageLoc():
    return os.path.join(configLoc(), "archive")


if __name__ == '__main__':
    print(configLoc())
    print(configImageLoc())