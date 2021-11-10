from __future__ import division     # my version is python2.7
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork:

    def __init__(self, sizes, batch=1024, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.batch = batch
        self.epochs = epochs
        self.l_rate = l_rate

        self.image_size = 28
        self.num_images_train = int(60000 * 0.7)
        self.num_images_valid = int(60000 * 0.3)
        self.num_images_test = 10000
        self.data = []
        self.label = []
        self.raw_label = []

        self.sum_grad = np.zeros(10)
        self.final_grad = np.zeros(10)

        self.params = self.init_params()    # why can return local variable ? (c.f. C++)
    

    def init_params(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_layer = self.sizes[1]
        hidden_layer2 = self.sizes[2]
        output_layer = self.sizes[3]

        # using dictionary to store parameter
        params = {
        'W1':np.random.randn(hidden_layer, input_layer) * np.sqrt(1. / hidden_layer),       # why multiply sqrt...?
        'W2':np.random.randn(hidden_layer2, hidden_layer) * np.sqrt(1. / hidden_layer2),
        'W3':np.random.randn(output_layer, hidden_layer2) * np.sqrt(1. / output_layer)
        }

        return params


    def read_training_data(self):
        # training image
        f = gzip.open('train-images-idx3-ubyte.gz','r')
        f.read(16)  # magic number
        buf = f.read(self.image_size * self.image_size * self.num_images_train)
        self.data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        self.data = self.data.reshape(self.num_images_train, self.image_size, self.image_size)

        # validation image
        buf = f.read(self.image_size * self.image_size * self.num_images_valid)
        self.data_valid = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        self.data_valid = self.data_valid.reshape(self.num_images_valid, self.image_size, self.image_size)

        # training label
        f = gzip.open('train-labels-idx1-ubyte.gz','r')
        f.read(8)
        buf = f.read(self.num_images_train)
        self.raw_label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)    # is already an array

        # validation label
        buf = f.read(self.num_images_valid)
        self.raw_label_valid = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)    # is already an array
    

    def read_testing_data(self):
        # testing image
        f = gzip.open('t10k-images-idx3-ubyte.gz','r')
        f.read(16)  # magic number
        buf = f.read(self.image_size * self.image_size * self.num_images_test)
        self.data_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        self.data_test = self.data_test.reshape(self.num_images_test, self.image_size, self.image_size)

        # validation image
        f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
        f.read(8)
        buf = f.read(self.num_images_test)
        self.raw_label_test = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)    # is already an array


    # convert the label into one-hot (i.e. [0 0 1 0 0 0...])
    def one_hot_label(self, label):
        for i in range(10):
            if(i == label):
                zero_arr = np.zeros(10)
                zero_arr[i] = 1
                return zero_arr

            
    def forward_pass(self, x_train):
        params = self.params

        params['A0'] = x_train / 255

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.relu(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.relu(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])
        
        # print("Z3")
        # print(params['Z3'])
        # print("A3")
        # print(params['A3'])

        return params['A3']
    

    def backpropagation(self, label, output):
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = self.final_grad.reshape(10,1)
        # print(error)
        change_w['W3'] = np.outer(error, params['A2'])  # partial_cw = partial_cz * partial_zw

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.relu(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.relu(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])
        
        # print(change_w)
        return change_w
    

    # update the weight
    def update_w(self, change_w):
        # print(self.params['W1'].shape)
        # print(change_w['W1'].shape)
        for key, value in change_w.items():
            self.params[key] -= self.l_rate * value


    # gradient of softmax + cross entropy
    def softmax_CrossEntropy_grad(self, label, softmax_z):
        grad_list = [(softmax_z[i] * np.sum(label) - label[i]) for i in range(10)]
        grad = np.asarray(grad_list).reshape(10)
        # print("grad: ")
        # print(grad)
        return grad


    # sum the gradient when mini batch SGD (for later average)
    def mini_batch_grad(self, grad):
        self.sum_grad += grad
        # print(self.sum_grad)


    # cross entropy loss
    def CE_loss(self, label, final_a):
        loss = 0.0
        for i in range(10):
            loss += label[i] * np.log(final_a[i])
        return - loss[0]    # need to calculate mean?


    def relu(self, x, derivative=False):
        if derivative:
            for i in range(x.size):
                if x[i] < 0: x[i] = 0   # call by reference? call by value?
                else: x[i] = 1
        for i in range(x.size):
            if x[i] < 0: x[i] = 0

        return x
    

    def softmax(self, x):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)


    # count the number of correct inference
    def correct(self, label, output):
        if(np.argmax(label) == np.argmax(output)):
            # print("label: ", (np.argmax(label)))
            # print("output: ", (np.argmax(output)))
            return 1
        else:
            return 0
    

    # training model using SGD (including validation)
    def training_SGD(self):
        print("start training !")
        self.read_training_data()
        plot_train_x = []
        plot_train_y = []
        plot_valid_x = []
        plot_valid_y = []
        for j in range(self.epochs):
            loss = 0
            num_correct = 0
            valid_loss = 0
            valid_num_correct = 0
            for i in range(self.num_images_train):
                self.sum_grad = np.zeros(10)
                self.final_grad = np.zeros(10)
                
                self.label = self.one_hot_label(self.raw_label[i])  # [0 0 0 1 0 0 ...]
                output = self.forward_pass(self.data[i].reshape(784, 1))
                
                loss += self.CE_loss(self.label, output)
                num_correct += self.correct(self.label, output)

                self.mini_batch_grad(self.softmax_CrossEntropy_grad(self.label, output))
                self.final_grad = self.sum_grad

                change_w = self.backpropagation(self.label, output)
                self.update_w(change_w)

                # validation
                valid_sample = random.randint(0, self.num_images_valid-1)
                valid_label = self.one_hot_label(self.raw_label_valid[valid_sample])  # [0 0 0 1 0 0 ...]
                valid_output = self.forward_pass(self.data_valid[valid_sample].reshape(784, 1))
                valid_loss += self.CE_loss(valid_label, valid_output)
                valid_num_correct += self.correct(valid_label, valid_output)


            # print("correct: ", num_correct)
            print("epoch " + str(j+1) + " - loss: " + str(loss / self.num_images_train)+ ", accuracy: " + str(num_correct / self.num_images_train))
            # print("epoch " + str(j+1) + " accuracy: " + str(num_correct / self.num_images_train))

            # print("valid_correct: ", valid_num_correct)
            print("epoch " + str(j+1) + " - valid_loss: " + str(valid_loss / self.num_images_train) + ", valid_accuracy: " + str(valid_num_correct / self.num_images_train))
            # print("epoch " + str(j+1) + " valid_accuracy: " + str(valid_num_correct / self.num_images_train))

            plot_train_x.append(j+1)
            plot_train_y.append(loss / self.num_images_train)
            plot_valid_x.append(j+1)
            plot_valid_y.append(valid_loss / self.num_images_train)

        print('end of training')
        plots = plt.plot(plot_train_x, plot_train_y, plot_valid_x, plot_valid_y)
        plt.legend(plots, ('training loss', 'validation loss'))
        plt.show()


    # training model using mini batch SGD, but haven't successed
    def training_mini_batch_SGD(self):
        self.read_training_data()
        for k in range(self.epochs):
            loss = 0
            num_correct = 0
            for j in range(int(self.num_images_train / self.batch)):
                sample = random.sample(range(1, self.num_images_train+1), self.batch)
                # print(sample)
                self.sum_grad = np.zeros(10)
                self.final_grad = np.zeros(10)
                for i in range(self.batch):
                    self.label = self.one_hot_label(self.raw_label[sample[i]-1])  # [0 0 0 1 0 0 ...]

                    output = self.forward_pass(self.data[sample[i]-1].reshape(784, 1))
                    # print("output:")
                    # print(output)
                    loss += self.CE_loss(self.label, output)
                    num_correct += self.correct(self.label, output)

                    self.mini_batch_grad(self.softmax_CrossEntropy_grad(self.label, output))    #accumulate grad
                
                # print("sum: ")
                # print(self.sum_grad)
                self.final_grad = self.sum_grad / self.batch
                # print("final: ")
                # print(self.final_grad)
                change_w = self.backpropagation(self.label, output)
                # print("change_w:")
                # print(change_w)

                self.update_w(change_w)

                # print("label: ")
                # print(self.label)
                # print("output: ")
                # print(output)

            print("correct: ", num_correct)
            print("epoch " + str(k+1) + " loss: " + str(loss / self.num_images_train))
            print("epoch " + str(k+1) + " accuracy: " + str(num_correct / self.num_images_train))

        # print(data[0])
        # print(data[0][0].shape)
        # print('data:', data[0][15][15])     # 3-d array
        # print(data[0][15][15])              # what's the type ? numpy.ndarray
        # print(int(data[0][15][15]))
        print('end of training_mini_batch_SGD')


    def testing(self):
        print("-")
        print("start testing !")
        self.read_testing_data()
        test_loss = 0
        test_num_correct = 0
        for i in range(self.num_images_test):
            test_label = self.one_hot_label(self.raw_label_test[i])  # [0 0 0 1 0 0 ...]
            test_output = self.forward_pass(self.data_test[i].reshape(784, 1))
            # test_loss += self.CE_loss(test_label, test_output)
            test_num_correct += self.correct(test_label, test_output)
        
        # print("test_correct: ", test_num_correct)
        # print("test_loss: " + str(valid_loss / self.num_images_train))
        print("test_accuracy: " + str(test_num_correct / self.num_images_test))
        print('end of testing')



if __name__ == '__main__':
    NN = NeuralNetwork([784, 500, 300, 10])     # the number is each layer's nodes
    NN.training_SGD()
    NN.testing()