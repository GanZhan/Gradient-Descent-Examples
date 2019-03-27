import numpy as np
from GradientDescent.data_utils import get_points

def current_loss_4_SGD(w_current, b_current, x, y, seed):
    y_pre = w_current*x[seed] + b_current
    loss = (y_pre - y[seed])**2
    return loss/float(1)

def step_gradient(w_current, b_current, x, y, lr, seed):
    w_gradient = 0
    b_gradient = 0
    w_gradient = w_gradient + (w_current*x[seed] + b_current - y[seed])*x[seed]
    b_gradient = b_gradient + (w_current*x[seed] + b_current - y[seed])
    w_current = w_current - lr*w_gradient/float(1)
    b_current = b_current - lr*b_gradient/float(1)
    return w_current, b_current

def gradient_descent_runner(w_starter, b_starter, x, y, _length, lr, steps):
    w_current = w_starter
    b_current = b_starter
    for i in range(steps):
        seed = np.random.choice(_length)
        w_current, b_current = step_gradient(w_current, b_current, x, y, lr, seed)
    return w_current, b_current

def main():
    lr = 0.0001
    w_starter = 0.001
    b_starter = 0.001
    steps = 2000
    filepath = 'data.csv'
    x, y, _length = get_points(filepath)
    seed = np.random.choice(_length)  # using same sample to watch change of loss
    #print(seed)
    print("Starting Stochastic gradient descent at w = {0}, b = {1}, loss = {2}".format(w_starter, b_starter,
                                                current_loss_4_SGD(w_starter, b_starter, x, y, seed)))
    print("Running...")
    w_current, b_current = gradient_descent_runner(w_starter, b_starter, x, y, _length, lr, steps)
    print("After {0} iterations w = {1}, b = {2}, loss = {3}".format(steps, w_current, b_current,
                                                current_loss_4_SGD(w_current, b_current, x, y, seed)))

if __name__ == '__main__':
    main()