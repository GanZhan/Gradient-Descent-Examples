import random
from GradientDescent.data_utils import get_points

def current_loss_4_MBGD(w_current, b_current, x, y, seed_list):
    loss = 0
    for seed in seed_list:
        loss = loss + (w_current*x[seed]+b_current - y[seed])**2
    return loss/float(len(seed_list))

def step_gradient(w_current, b_current, x, y, lr, seed_list):
    w_gradient = 0
    b_gradient = 0
    for seed in seed_list:
        w_gradient = w_gradient + (w_current*x[seed]+b_current - y[seed])*x[seed]
        b_gradient = b_gradient + (w_current*x[seed]+b_current - y[seed])
    w_current = w_current - lr*w_gradient/float(len(seed_list))
    b_current = b_current - lr*b_gradient/float(len(seed_list))
    return w_current, b_current

def gradient_descent_runner(w_starter, b_starter, x, y, _length, lr, steps, batch_size):
    w_current = w_starter
    b_current = b_starter
    seed_list = random.sample(range(_length), batch_size)
    for i in range(steps):
        w_current, b_current = step_gradient(w_current, b_current, x, y, lr, seed_list)
    return w_current, b_current

def main():
    lr = 0.0001
    w_starter = 0.0001
    b_starter = 0.0001
    steps = 2000
    filepath = 'data.csv'
    x, y, _length = get_points(filepath)
    batch_size = 4
    seed_list = random.sample(range(_length), batch_size)  # using the batch of samples to watch the change of loss
    #print(seed_list)
    print("Starting mini-batch gradient descent at w = {0}, b = {1}, loss = {2}".format(w_starter, b_starter,
                                                                             current_loss_4_MBGD(w_starter, b_starter, x,
                                                                                                y, seed_list)))
    print("Running...")
    w_current, b_current = gradient_descent_runner(w_starter, b_starter, x, y, _length, lr, steps, batch_size)
    print("After {0} iterations w = {1}, b = {2}, loss = {3}".format(steps, w_current, b_current,
                                                                     current_loss_4_MBGD(w_current, b_current, x, y,
                                                                                        seed_list)))

if __name__ == '__main__':
    main()