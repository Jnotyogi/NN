# NNJ 1 or 2 hidden layers
import numpy as np
J = 6 ; M = 6   # numbers of nodes in the hidden layers 1 and 2 (set M to zero for just one hidden layer)

def sigmoid(f):    # but other activation functions can be tried
    return 1.0/(1+ np.exp(-f))

def sigmderiv(z):    # derivative of above function but note that z is the value sigmoid(f) not the sigmoid argument f
    return z * (1.0 - z)

class NeuralNetwork:
    def __init__(self, x, y):
        if M > 0:       # ie if there is a second hidden layer 
            self.z3ea  = x    # input 
            self.w32am = np.random.rand(A,M) * 0.1 # initial weights for layer 3 to layer 2 (hidden)
            self.w21mj = np.random.rand(M,J) * 0.1 # initial weights for layer 2 to layer 1 (hidden)
        else:
            self.z2em  = x    # input although m is a now !! (when M=0)
            self.w21mj = np.random.rand(A,J) * 0.1 # initial weights for layer 2 to layer 1 (hidden)
        self.w10jp     = np.random.rand(J,P) * 0.1 # initial weights for layer 1 to layer 0 (output)
        self.t0ep      = y    # Target training results of e training cases for layer 0 (output): p values each

    def feedforward(self): 
        if M > 0:
            self.z2em = sigmoid(np.dot(self.z3ea, self.w32am))    # = sigmoid(f2em)
        self.z1ej = sigmoid(np.dot(self.z2em, self.w21mj))        # = sigmoid(f1ej)
        self.z0ep = sigmoid(np.dot(self.z1ej, self.w10jp))        # = sigmoid(f0ep)

    def backprop(self):
        dEp = (self.t0ep - self.z0ep) * sigmderiv(self.z0ep)       # = -dE/df0ep
        self.w10jp += np.dot(self.z1ej.T, dEp)    
        
        dEj = np.dot(dEp, self.w10jp.T) * sigmderiv(self.z1ej)     # = -de/df1ej
        self.w21mj += np.dot(self.z2em.T, dEj)
        
        if M > 0:
            dEm = np.dot(dEj,self.w21mj.T)*sigmderiv(self.z2em)    # = -dE/df2em
            self.w32am += np.dot(self.z3ea.T, dEm)

    def singlerun(self,zInput): 
        if M > 0: 
            z2m = sigmoid(np.dot(zInput, self.w32am))
            z1j = sigmoid(np.dot(z2m, self.w21mj))
        else:
            z1j = sigmoid(np.dot(zInput, self.w21mj))
        return sigmoid(np.dot(z1j, self.w10jp))


# main
# input defines by E down and A across, the E examples of the nodes of the input layer

TrainingInput_ea =  np.array([
                    [0,1,1,1,0,0],
                    [0,1,1,1,0,1],
                    [0,1,1,1,1,0],
                    [1,1,1,1,0,0],
                    [0,1,1,0,1,1],
                    [0,0,1,1,1,0],

                    [0,0,0,1,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0],
                    [1,0,0,0,0,0]])

TrainingOutput_ep = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],    [0,1],[0,1],[0,1],[0,1]])

E = TrainingInput_ea.shape[0]
A = TrainingInput_ea.shape[1]
P = TrainingOutput_ep.shape[1]

nn = NeuralNetwork(TrainingInput_ea,TrainingOutput_ep)    # define class and its initial setup with random weights

T = 9000    # train with this number of training sets each employing all of the E example/result combos
for ts in range(T):    
    nn.feedforward() # get output result for this set, starting with weights from previous set
    nn.backprop() # modify the weighting factors following this set

print("A M J P T :  ", A, M, J, P, T)
print("From training:")
print(nn.z0ep)

# Single runs each with their own input data: print target then result
print("New cases:")
print(1,0,nn.singlerun(np.array([0,1,1,1,0,0]))) # same as training case
print(0,1,nn.singlerun(np.array([0,0,1,0,0,0]))) # same as training case
print("u",nn.singlerun(np.array([0,0,0,0,0,1]))) # unspecified as not one of the training cases
print()
