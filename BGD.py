import numpy as np
from GradientDescent.data_utils import get_points

def current_loss_4_BGD(w_current, b_current, x, y, _length):
    """
    calculate current loss for batch gradient descent at every step: (w_current, b_current)
    :param w_current:
    :param b_current:
    :param x:
    :param y:
    :param _length:
    :return:
    """
    total_loss = 0
    for i in range(_length):
        y_pre = w_current*x[i] + b_current
        total_loss = total_loss + (y[i]-y_pre)**2   # using least square to get the loss   (np.square)
    # average total loss
    return total_loss/ float(_length)

def step_gradient(w_current, b_current, x, y, _length, lr):
    """
    calculate gradient of current w & b in one step, then update them to get new w_current & b_current 
    :param w_current:
    :param b_current:
    :param x:
    :param y:
    :param _length:
    :param lr:
    :return:
    """
    w_gradient = 0
    b_gradient = 0
    for i in range(_length):
        x_i = x[i]
        y_i = y[i]
        w_gradient += (w_current*x_i + b_current-y_i)*x_i
        b_gradient += (w_current*x_i + b_current-y_i)
    w_current = w_current -lr*w_gradient/float(_length)
    b_current = b_current -lr*b_gradient/float(_length)
    return w_current, b_current

def gradient_descent_runner(w_starter, b_starter, x, y, _length, lr, steps):
    """
    calculate the value for w_current & b_current after certain BGD steps
    :param w_starter:
    :param b_starter:
    :param x:
    :param y:
    :param _length:
    :param lr:
    :param steps:
    :return:
    """
    w_current = w_starter
    b_current = b_starter
    for i in range(steps):
        w_current, b_current = step_gradient(w_current, b_current, x, y, _length, lr)
    return w_current, b_current


def main():
    """
    given a bunch of data, using different geadient descent algorithms to do the linear regression
    calculate the total loss at first with w_starter & b_starter
    train it with BGD for certain steps
    then calculate the loss for current W and b
    :return:
    """
    filepath = 'data.csv'
    x, y, _length = get_points(filepath)
    steps = 2000
    lr = 0.0001
    b_starter = 0
    w_starter = 0
    print("Starting batch gradient descent at w = {0}, b = {1}, loss = {2}".format(w_starter, b_starter,
                                                current_loss_4_BGD(w_starter, b_starter, x, y, _length)))
    print("Running...")
    w_current, b_current = gradient_descent_runner(w_starter, b_starter, x, y, _length, lr, steps)
    print("After {0} iterations w = {1}, b = {2}, loss = {3}".format(steps, w_current, b_current,
                                                current_loss_4_BGD(w_current, b_current, x, y, _length)))

if __name__ == '__main__':
    main()
