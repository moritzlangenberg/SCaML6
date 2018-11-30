from network import *

print("\nLoading Iris test data ")
testDataPath = "irisTestData.txt"
testDataset = IrisDataset(testDataPath)

print("\nLoading Iris Training data ")
trainDataPath = "irisTrainData.txt"
trainDataset = IrisDataset(trainDataPath)

net = BasicNeuralNetwork()

net.train(train_dataset=trainDataset, eval_dataset=testDataset)
net.load('./task_2a.save')
acc = net.accuracy(testDataset)

#some added tests
print(str(acc))
#####END#########

if acc > 0.5:
    print('Forwarding seems to function correctly')
else:
    print('There seem to be errors with your computation of the network outputs')
exit()

