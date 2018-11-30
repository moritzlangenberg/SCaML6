import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


class Dataset:
    def __init__(self):
        self.index = 0

        self.obs = []
        self.classes = []
        self.num_obs = 0
        self.num_classes = 0
        self.indices = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_obs:
            self.index = 0
            raise StopIteration
        else:
            self.index += 1
            return self.obs[self.index - 1], self.classes[self.index - 1]

    def reset(self):
        self.index = 0

    def get_obs_with_target(self, k):
        index_list = [index for index, value in enumerate(self.classes) if value == k]
        return [self.obs[i] for i in index_list]

    def get_all_obs_class(self, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)
        return [(self.obs[i], self.classes[i]) for i in self.indices]

    def get_mini_batches(self, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)

        batches = [(self.obs[self.indices[n:n + batch_size]],
                    self.classes[self.indices[n:n + batch_size]])
                   for n in range(0, self.num_obs, batch_size)]
        return batches


class IrisDataset(Dataset):
    def __init__(self, path):
        super(IrisDataset, self).__init__()
        self.file_path = path
        self.loadFile()
        self.indices = np.arange(self.num_obs)

    def loadFile(self):
        # load a comma-delimited text file into an np matrix
        resultList = []
        f = open(self.file_path, 'r')
        for line in f:
            line = line.rstrip('\n')  # "1.0,2.0,3.0"
            sVals = line.split(',')  # ["1.0", "2.0, "3.0"]
            fVals = list(map(np.float32, sVals))  # [1.0, 2.0, 3.0]
            resultList.append(fVals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
        f.close()
        data = np.asarray(resultList, dtype=np.float32)  # not necessary
        self.obs = data[:, 0:4]
        self.classes = data[:, 4:7]
        self.num_obs = data.shape[0]
        self.num_classes = 3

# Helpers
def square(list):
    return np.array([i ** 2 for i in list])


# Activations
def tanh(x, deriv=False):
    '''
	d/dx tanh(x) = 1 - tanh^2(x)
	during backpropagation when we need to go though
	the derivative we have already computed tanh(x),
	therefore we pass tanh(x) to the function which reduces the gradient to:
	1 - tanh(x)
    '''
    if deriv:
        return 1.0 - np.power(np.tanh(x), 2)
    else:
        return np.tanh(x)


def sigmoid(x, deriv=False):
    '''
    Task 2a
    This function is the sigmoid function.
    It gets an input digit or vector and should return sigmoid(x).
    The parameter "deriv" toggles between the sigmoid and the derivate of the sigmoid.
     Hint: In the case of the derivate
    you can expect the input to be sigmoid(x) instead of x
    :param x:
    :param deriv:
    :return:
    '''
    if deriv:
        return sigmoid(x, False) * (1 - sigmoid(x, False))
    else:
        if type(x) == 'float32':
            return [0.5 * (1 + tanh(0.5 * i)) for i in x]
        else:
            return 0.5 * (1 + tanh(0.5 * x))


def softmax(x, deriv=False):
    '''
    Task 2a
    This function is the sigmoid function with a softmax applied.
    This will be used in the last layer of the network
    The derivate will be the same as of sigmoid(x)
    :param x:
    :param deriv:
    :return:
    '''
    if deriv:
        return sigmoid(x, True)
    else:
        exps = [np.exp(i) for i in x]
        sumOfExps = np.sum(exps)
        return np.array([np.divide(i, sumOfExps) for i in exps])


class Layer:
    def __init__(self, numInput, numOutput, activation=sigmoid):
        print('Create layer with: {}x{} @ {}'.format(numInput, numOutput, activation))
        self.ni = numInput
        self.no = numOutput
        self.weights = np.zeros(shape=[self.ni, self.no], dtype=np.float32)
        self.biases = np.zeros(shape=[self.no], dtype=np.float32)
        self.initializeWeights()

        self.activation = activation
        self.last_input = np.zeros(shape=[self.ni], dtype=np.float32)
        self.last_output = np.zeros(shape=[self.no], dtype=np.float32)
        self.last_nodes = np.zeros(shape=[self.no], dtype=np.float32)

    def initializeWeights(self):
        """
        Task 2d
        Initialized the weight matrix of the layer.
        Weights should be initialized to something other than 0.
        You can search the literature for possible initialization methods.
        :return: None
        """
        self.weights = np.random.rand(self.ni, self.no)
        self.biases = np.random.rand(self.no)

    def inference(self, x):
        """
        Task 2b
        This transforms the input x with the layers weights and bias and applies the activation function
        Hint: you should save the input and output of this function usage in the backpropagation
        :param x:
        :return: output of the layer
        :rtype: np.array
        """
        self.last_input = x
        z = np.zeros(shape=[self.no], dtype=np.float32)
        for i in range(0, self.no):
            for j in range(0, self.ni):
                z[i] += self.weights[j][i] * x[j]
            z[i] += self.biases[i]
        self.last_nodes = z
        self.last_output = self.activation(z)
        return self.last_output

    def backprop(self, error):
        """
        Task 2c
        This function applied the backpropagation of the error signal.
        The Layer receives the error signal from the following
        layer or the network. You need to calculate the error signal
        for the next layer by backpropagating thru this layer.
         You also need to compute the gradients for the weights and bias.
        :param error:
        :return: error signal for the preceeding layer
        :return: gradients for the weight matrix
        :return: gradients for the bias
        :rtype: np.array
        """
        #Gradient for weights
        grad_weights = np.zeros(shape=[self.ni, self.no], dtype=np.float32)
        for i in range(0, self.no):
            for j in range(0, self.ni):
                grad_weights[j][i] = error[i] \
                                     * self.activation(self.last_nodes[i], True) \
                                     * self.last_input[j]

        #Gradient for biases
        grad_biases = np.zeros(shape=[self.no], dtype=np.float32)
        for i in range(0, self.no):
            grad_biases[i] = error[i] \
                             * self.activation(self.last_nodes[i], True)
        #Error signal for prev Layer E/y(l-1)
        errorSignal = np.zeros(shape=[self.ni], dtype=np.float32)
        for i in range(0, self.ni):
            for k in range(0, self.no):
                errorSignal[i] += error[k] \
                                  * self.activation(self.last_nodes[k], True) \
                                  * self.weights[i][k]
        return (errorSignal, grad_weights, grad_biases)


class BasicNeuralNetwork():
    def __init__(self, layer_sizes=5, num_input=4, num_output=3, num_epoch=50, learning_rate=0.1,
                 mini_batch_size=8, number_of_hiddenlayers=1):
        self.layers = []
        self.ls = layer_sizes
        self.ni = num_input
        self.no = num_output
        self.lr = learning_rate
        self.num_epoch = num_epoch
        self.mbs = mini_batch_size
        self.nhl = number_of_hiddenlayers
        self.constructNetwork()

        self.mbs = None

    def forward(self, x):
        """
        Task 2b
        This function forwards a single feature vector through every layer and
         return the output of the last layer
        :param x: input feature vector
        :return: output of the network
        :rtype: np.array
        """
        #self.constructNetwork()
        #y = np.zeros(shape=self.ls) --> accuracy = 33,3%
        y = self.layers[0].inference(x)
        for layerIterator in range(1, len(self.layers)):
            y = self.layers[layerIterator].inference(y)
            #print(str(y))
        return y

    def train(self, train_dataset, eval_dataset=None, monitor_ce_train=True, monitor_accuracy_train=True,
              monitor_ce_eval=True, monitor_accuracy_eval=True, monitor_plot='monitor.png'):
        ce_train_array = []
        ce_eval_array = []
        acc_train_array = []
        acc_eval_array = []
        for e in range(self.num_epoch):
            if self.mbs:
                self.mini_batch_SGD(train_dataset)
            else:
                self.online_SGD(train_dataset)
            print('Finished training epoch: {}'.format(e))
            if monitor_ce_train:
                ce_train = self.ce(train_dataset)
                ce_train_array.append(ce_train)
                print('CE (train): {}'.format(ce_train))
            if monitor_accuracy_train:
                acc_train = self.accuracy(train_dataset)
                acc_train_array.append(acc_train)
                print('Accuracy (train): {}'.format(acc_train))
            if monitor_ce_eval:
                ce_eval = self.ce(eval_dataset)
                ce_eval_array.append(ce_eval)
                print('CE (eval): {}'.format(ce_eval))
            if monitor_accuracy_eval:
                acc_eval = self.accuracy(eval_dataset)
                acc_eval_array.append(acc_eval)
                print('Accuracy (eval): {}'.format(acc_eval))

        if monitor_plot:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            line1, = ax[0].plot(ce_train_array, '--', linewidth=2, label='ce_train')
            line2, = ax[0].plot(ce_eval_array, label='ce_eval')

            line3, = ax[1].plot(acc_train_array, '--', linewidth=2, label='acc_train')
            line4, = ax[1].plot(acc_eval_array, label='acc_eval')

            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper left')
            ax[1].set_ylim([0, 1])

            plt.savefig(monitor_plot)

    def online_SGD(self, dataset):
        """
        Task 2d
        This function trains the network in an online fashion.
        Meaning the weights are updated after each observation.
        :param dataset:
        :return: None
        """
        for (o, k) in dataset.get_all_obs_class():
            temp = self.forward(o)
            error = np.absolute(temp - k)
            #print(k)
            #print(temp)
            #print(temp-k)
            #print(error)
            print(error)
            #print(k)
            for layer in reversed(self.layers):
                (e, w, b) = layer.backprop(error)
                layer.weights = layer.weights - self.lr * w
                layer.biases = layer.biases - self.lr * b
                error = None
                error = e

    def mini_batch_SGD(self, dataset):
        """
        Task 2d
        This function trains the network using mini batches.
        Meaning the weights updates are accumulated and applied after each mini batch.
        :param dataset:
        :return: None
        """
        """
            def get_mini_batches(self, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)

        batches = [(self.obs[self.indices[n:n + batch_size]],
                    self.classes[self.indices[n:n + batch_size]])
                   for n in range(0, self.num_obs, batch_size)]
        return batches
        """
        for (o, k) in dataset.get_mini_batches(self.mbs):
            error = np.zeros(shape=[self.no])
            for b in range(0, self.mbs):
                error += square(self.forward(o[b]) - k[b])
                #error[int(k[b])] += (1 - 2 * currError[int(k[b])])
            for layer in reversed((self.layers)):
                (e, w, b) = layer.backprop(error)
                layer.weights -= self.lr * w
                layer.weights -= self.lr * b
                error = e

    def constructNetwork(self):
        """
        Task 2d
        uses self.ls self.ni and self.no to construct a list of layers.
        The last layer should use sigmoid_softmax as an activation function.
        any preceeding layers should use sigmoid.
        :return: None
        """
        input_layer = Layer(self.ni, self.ls, sigmoid)
        self.layers = [input_layer]
        for i in range(0, self.nhl):
            hidden_layer = Layer(self.ls, self.ls, sigmoid)
            self.layers.append(hidden_layer)
        output_layer = Layer(self.ls, self.no, softmax)
        self.layers.append(output_layer)



    def ce(self, dataset):
        ce = 0
        for x, t in dataset:
            t_hat = self.forward(x)
            ce += np.sum(np.nan_to_num(-t * np.log(t_hat) - (1 - t) * np.log(1 - t_hat)))

        return ce / dataset.num_obs


    def accuracy(self, dataset):
        cm = np.zeros(shape=[dataset.num_classes, dataset.num_classes], dtype=np.int)
        for x, t in dataset:
            t_hat = self.forward(x)
            c_hat = np.argmax(t_hat)  # index of largest output value
            c = np.argmax(t)
            cm[c, c_hat] += 1

        correct = np.trace(cm)
        return correct / dataset.num_obs


    def load(self, path=None):
        if not path:
            path = './network.save'
        with open(path, 'rb') as f:
            self.layers = pickle.load(f)


    def save(self, path=None):
        if not path:
            path = './network.save'
        with open(path, 'wb') as f:
            pickle.dump(self.layers, f)
