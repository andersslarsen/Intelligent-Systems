import numpy as np
import torch as T
import random
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
device = T.device("cpu")

class CancerDataset(T.utils.data.Dataset):
    def __init__(self, src_file, n_rows=None):
        allCols = np.loadtxt(src_file, max_rows=n_rows,
          usecols=[1,2,3,4,5,6,7,8,9,10], delimiter=",",dtype=np.float32)

        n = len(allCols)
        x_attr = allCols[0:n,0:9]
        y_class = allCols[0:n,9]

        self.x = T.tensor(x_attr, dtype=T.float32).to(device)
        self.y = T.tensor(y_class, dtype=T.int64).to(device)

    def __len__(self):
          return len(self.x)

    def __getitem__(self, idx):
        preds = self.x[idx]
        trgts = self.y[idx]
        sample = {
            'predictors' : preds,
            'targets' : trgts
          }
        return sample


class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    #Input, hidden, output layers with
    # 9, 10 and 5 nodes respectively
    self.inLayer = T.nn.Linear(9, 10)
    self.hid = T.nn.Linear(10, 10)
    self.out = T.nn.Linear(10, 5)

    #weight and bias initialization
    T.nn.init.xavier_uniform_(self.inLayer.weight)
    T.nn.init.zeros_(self.inLayer.bias)
    T.nn.init.xavier_uniform_(self.hid.weight)
    T.nn.init.zeros_(self.hid.bias)
    T.nn.init.xavier_uniform_(self.out.weight)
    T.nn.init.zeros_(self.out.bias)

  def forward(self, x):
    z = T.tanh(self.inLayer(x))
    z = T.tanh(self.hid(z))
    z = self.out(z)
    return z

#Function which calculates the accuracy of the model
def accuracy(model, dataset):
    numCor = 0; numWro = 0
    for i in range(len(dataset)):
        X = dataset[i]['predictors']
        Y = dataset[i]['targets'] ## [2] or [4]
        with T.no_grad():
            out = model(X)
        index = T.argmax(out)  ## [2] or [4]
        if index == Y:
            numCor += 1
        else:
            numWro += 1
    acc = (numCor * 1.0) / (numCor + numWro)
    return acc

#Function which splits the original dataset into training and test sets
def splitDataset():
    orig_file = open("cancer.csv")
    orig_arr = np.loadtxt(orig_file, delimiter=",")
    data_train, data_test = train_test_split(orig_arr)
    np.savetxt('train.csv', data_train, delimiter=',', fmt='%.0f')
    np.savetxt('test.csv', data_test, delimiter=',', fmt='%.0f')
    train_length = len(data_train)
    test_length = len(data_test)
    return train_length, test_length

#Converts the generated test.csv-file into an array
def setsToArrays():
    test_file = open("test.csv")
    testArray = np.loadtxt(test_file,usecols=[1,2,3,4,5,6,7,8,9], delimiter=",")
    return testArray

def loss_plot(loss, ep):
    max = 10
    new_loss = []
    count = 0
    sum = 0
    for i in range(len(loss)):
        count += 1
        sum += loss[i]
        if count == max:
            new_loss.append(sum/max)
            count = 0
            sum = 0
    loss = new_loss
    x_axis = np.linspace(0, ep, len(loss))
    plt.plot(x_axis, loss, label="training loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc='best')
    plt.show()

def accuracy_plot(accuracy_train, ep):
    x_axis = np.linspace(0, ep, len(accuracy_train))
    y_axis = np.linspace(0, 1, len(accuracy_train))
    plt.plot(x_axis, accuracy_train, label="training accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='best')
    plt.show()


def main():

    splitData = splitDataset()
    train_length = splitData[0]
    test_length = splitData[1]

    train_file = "train.csv"
    train_dataset = CancerDataset(train_file, n_rows=train_length)

    test_file = "test.csv"
    test_dataset = CancerDataset(test_file, n_rows=test_length)

    bat_size = 10
    train_ldr = T.utils.data.DataLoader(train_dataset, batch_size=bat_size, shuffle=True)

    net = Net().to(device)

    max_epochs = 100
    ep_interval = 5
    lrn_rate = 0.01

    loss_func = T.nn.CrossEntropyLoss()
    optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

    net.train()

    epoch_loss_array = []
    acc_list = []
    for epoch in range(0, max_epochs):
        epoch_loss = 0
        #returns a tuple with the current batch
        #zero-based index value, and the actual batch of data
        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors']
            Y = batch['targets']

            optimizer.zero_grad()
            out = net(X)

            loss_val = loss_func(out, Y)  # avg loss in batch
            epoch_loss += loss_val.item()  # a sum of averages
            epoch_loss_array.append(epoch_loss)
            loss_val.backward()
            optimizer.step()

        if epoch % ep_interval == 0:
          print("------------------------------------")
          print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
        acc_ep = accuracy(net, train_dataset)
        acc_list.append(acc_ep)
    plotLoss = loss_plot(epoch_loss_array, max_epochs)
    plotAcc = accuracy_plot(acc_list, max_epochs)

    # Evaluating the model's accuracy
    print("\nComputing model accuracy")
    net.eval()
    acc_test = accuracy(net, test_dataset)  # en masse
    print("Accuracy on test data = %0.4f" % acc_test)

    # Create a matrix/array with every
    # prediction and convert to .csv file
    test_array = setsToArrays()
    inpt = T.tensor(test_array, dtype=T.float32).to(device)
    with T.no_grad():
        logits = net(inpt)      # values do not sum to 1.0
    probs = T.softmax(logits, dim=1)  # tensor
    probs = probs.numpy()
    np.set_printoptions(precision=4, suppress=True)

    results = []
    for values in range(len(probs)):
        result = max(probs[values])
        index_result = np.where(probs[values]==result)
        results.append(index_result[0][0])
    np.savetxt('results.csv', results, fmt='%.0f')

    T.save(net.state_dict(), "model.pt")
    model = Net()
    model.load_state_dict(T.load("model.pt"))
    model.eval()

if __name__ == "__main__":
  main()
