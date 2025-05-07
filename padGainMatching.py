import numpy as np
import matplotlib.pyplot as plt
# plt.style.use(['ggplot'])

# create calibration array for all 1020 pads
pad_gains = np.full((1021,2),1) # one extra for the overall params (scale and offset) to go from charge to energy
pad_gains = np.full((2,2),1) # for my test with 2 pads

# read in data
#test assuming 2 pads
ysum = 0.3 * np.random.randn(100000,1) + 6.288 # we know the total energy of each event
y1 = np.empty_like(ysum)
y2 = np.empty_like(ysum)
x1 = np.empty_like(ysum)
x2 = np.empty_like(ysum)
for i in range(len(ysum)):
    y1[i] = np.random.random() * ysum[i] # energy deposited in pad 1, anywhere from none to the all of the energy on the pad 
    y2[i] = ysum[i] - y1[i] # energy deposited in pad 2
    x1[i] = 100 + 10000 * y1[i] # charge deposited on pad 1, using arbitrary but known gain values
    x2[i] = -50 + 9000 * y2[i] # total charge must be conserved, with another set of arbitrary but known gain values



def calc_cost(params,x,y):
    m = len(y)
    predictions = np.dot(x,params).reshape(len(x[:,0]),1)
    cost = 0.5*m*np.sum(np.square(predictions-y))
    return cost

def gradient_descent(x,y,params,learning_rate=0.01,iterations=100):
    # parameters take the form [[offset1], [gain1], [offset2], [gain2], ...] for all pads
    params = params.flatten()
    params = params.reshape(len(params),1)

    # adding extra columns 
    for i in range(2*len(x[0,:])):
        if i==0:
            x_sub = np.ones((len(x[:,0])))
            continue
        if i % 2:
            x_sub = np.c_[x_sub,x[:,int((i-1)/2)]]
        else:
            x_sub = np.c_[x_sub,np.ones((len(x[:,0])))]
    
    x_sum = np.full([len(x[:,0])],1)
    x_sum = np.c_[x_sum,np.sum(x,axis=1)]
    m=len(y) # number of data points you have 
    cost_history=np.zeros(iterations)
    param_history=np.zeros((iterations,len(params)))
    print(x_sub)
    print(params)
    previous_variance = np.inf
    for i in range(iterations):
        prediction = np.dot(x_sub,params).reshape(len(x[:,0]),1)
        # print("x vals shape: ",np.shape(x_sub))
        # print("y vals shape: ",np.shape(y))
        # print("y vals prediction: ",np.shape(prediction))
        # print("params shape: ",np.shape(params))
        # print("x dot pred. - y shape: ",np.shape(x_sub.T.dot((prediction-y))))
        current_variance = np.var(prediction-y)
        current_mean = np.mean(prediction)
        params = params - (1/m)*learning_rate*(x_sub.T.dot((prediction-y)))
        params = params - (1/m)*learning_rate*(derivative of )
        if i == 0:
            print(prediction)
            print(np.shape(y))
            print(np.shape(prediction-y))
            print(np.shape((1/m)*learning_rate*x_sub.T.dot(prediction - y)))
        # if not i % 10:
        #     print("x dot pred. - y: ",x_sub.T.dot((prediction-y)))
        #     print("Iteration and Current Parameters: ", i, params)
        # params[2:] = params[2:] - (1/m)*learning_rate*(x_sub.T.dot((prediction_sub-y)).reshape(4,))
        # params[:2] = params[:2] - (1/m)*learning_rate*(x_sum.T.dot((prediction_sum-y)).reshape(2,))
        param_history[i,:]=params.flatten().T
        cost_history[i] = calc_cost(params,x_sub,y)
    return params, cost_history, param_history

lr = 0.01
n_iter = 1000
X_b = np.c_[x1,x2]
yenergy = np.full((100000,1),6.288)
theta,cost_history,theta_history = gradient_descent(X_b,yenergy,pad_gains,lr,n_iter)


# print(theta)
# print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
# print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

# fig,ax = plt.subplots(figsize=(12,8))

# ax.set_ylabel('J(Theta)')
# ax.set_xlabel('Iterations')
# _=ax.plot(range(n_iter),cost_history,'b.')
# plt.show()
