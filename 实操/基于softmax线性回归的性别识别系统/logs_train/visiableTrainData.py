import re


def get_data(fileName):
    cnt_list = []
    loss_list = []
    sum_loss_list = []
    accuracy_list = []

    with open(fileName) as f:
        for line in f:
            cnt_match = re.search("cnt==(\d+)", line)
            if cnt_match:
                cnt = int(cnt_match.group(1))
                cnt_list.append(cnt)

            loss_match = re.search("loss==(\d+\.\d+)", line)
            if loss_match:
                loss = float(loss_match.group(1))
                loss_list.append(loss)

            sum_loss_match = re.search("sum_loss:(\d+\.\d+)", line)
            if sum_loss_match:
                sum_loss = float(sum_loss_match.group(1))
                sum_loss_list.append(sum_loss)

            accuracy_match = re.search("test accuracy: (\d+\.\d+)", line)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                accuracy_list.append(accuracy)
    # epoch_list = [i for i in range(len(accuracy_list))]

    return cnt_list, loss_list, sum_loss_list, accuracy_list


Adam = get_data("logs optimer:Adam,lr=0.01.txt")
Lion = get_data("logs optimer:Lion.txt")
SGD = get_data("logs optimer:SGD,lr=0.01.txt")

import matplotlib.pyplot as plt

ID = 3
epochsA = len(Adam[ID])
epochsB = len(Lion[ID])
epochsC = len(SGD[ID])
plt.plot(range(epochsA), Adam[ID], label='Adam')
plt.plot(range(epochsB), Lion[ID], label='Lion')
plt.plot(range(epochsB), SGD[ID], label='SGD')

plt.title('Accuracy Comparison')
plt.xlabel('test times')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("end")