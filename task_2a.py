from network import *

print("\nLoading Iris test data ")
testDataPath = "irisTestData.txt"
testDataset = IrisDataset(testDataPath)

print("\nLoading Iris Training data ")
trainDataPath = "irisTrainData.txt"
trainDataset = IrisDataset(trainDataPath)

net = BasicNeuralNetwork()



net.train(train_dataset=trainDataset, eval_dataset=testDataset)

#####END#########
net.load('./task_2a.save')
acc = net.accuracy(testDataset)
net.save()
#some added tests
print(str(acc))
if acc > 0.5:
    print('Forwarding seems to function correctly')
else:
    print('There seem to be errors with your computation of the network outputs')
exit()

