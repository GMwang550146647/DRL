import numpy as np


def my_sigmoid(x, threshold):
    def sigmoid(x, up_down=0., scale_x=1., scale_y=1.):
        """
        :param x:
            x -> 03*scale -> -0.09
            x -> -2*scale -> -0.76
            x -> -scale -> -0.46
            x -> scale -> 0
            x -> scale  -> 0.46
            x -> 2*scale -> 0.76
            x -> 3*scale -> 0.9
        :param up_down: up or down
        :param scale_x: scale x
        :param scale_y: scale y
        :return:
        """
        x = x / scale_x
        x = np.clip(x, a_min=-25, a_max=25)
        return scale_y * (1 / (1 + np.exp(-x)) + up_down)

    def sigmoid_steps(x, step, level, division=10):
        y = np.zeros_like(x)
        for i in range(len(step) - 1):
            scale_x = (step[i + 1] - step[i]) / division
            scale_y = level[i + 1] - level[i]
            stepi = (step[i + 1] + step[i]) / 2
            yi_pos = sigmoid(x - stepi, up_down=0, scale_x=scale_x, scale_y=scale_y)
            yi_neg = sigmoid(x + stepi, up_down=-1, scale_x=scale_x, scale_y=scale_y)
            y = y + yi_pos + yi_neg
        return y

    steps = [0, threshold / 1.2, threshold, threshold * 3.5, threshold * 9]
    levels = [0, 0.1, 0.25, 0.5, 1]
    return sigmoid_steps(x, steps, levels)


def my_sigmoid_ori(x, threshold):
    def sigmoid(x, up_down=0., scale_x=1., scale_y=1.):
        """
        :param x:
            x -> 03*scale -> -0.09
            x -> -2*scale -> -0.76
            x -> -scale -> -0.46
            x -> scale -> 0
            x -> scale  -> 0.46
            x -> 2*scale -> 0.76
            x -> 3*scale -> 0.9
        :param up_down: up or down
        :param scale_x: scale x
        :param scale_y: scale y
        :return:
        """
        x = x / scale_x
        x = np.clip(x, a_min=-25, a_max=25)
        return scale_y * (1 / (1 + np.exp(-x)) + up_down)

    def sigmoid_steps(x, step, level, division=10):
        y = np.zeros_like(x)
        for i in range(len(step) - 1):
            scale_x = (step[i + 1] - step[i]) / division
            scale_y = level[i + 1] - level[i]
            stepi = (step[i + 1] + step[i]) / 2
            yi_pos = sigmoid(x - stepi, up_down=0, scale_x=scale_x, scale_y=scale_y)
            yi_neg = sigmoid(x + stepi, up_down=-1, scale_x=scale_x, scale_y=scale_y)
            y = y + yi_pos + yi_neg
        return y

    steps = [0, threshold / 1.5, threshold, threshold * 3, threshold * 8]
    levels = [0, 0.15, 0.3, 0.5, 1]
    return sigmoid_steps(x, steps, levels)


def my_signoid_m(x, threshold):
    def sigmoid(x, up_down=0., scale_x=1., scale_y=1.):
        """
        :param x:
            x -> 03*scale -> -0.09
            x -> -2*scale -> -0.76
            x -> -scale -> -0.46
            x -> scale -> 0
            x -> scale  -> 0.46
            x -> 2*scale -> 0.76
            x -> 3*scale -> 0.9
        :param up_down: up or down
        :param scale_x: scale x
        :param scale_y: scale y
        :return:
        """
        x = x / scale_x
        x = np.clip(x, a_min=-25, a_max=25)
        return scale_y * (1 / (1 + np.exp(-x)) + up_down)

    def sigmoid_steps(x, step, level, division=10):
        y = np.zeros_like(x)
        for i in range(len(step) - 1):
            scale_x = (step[i + 1] - step[i]) / division
            scale_y = level[i + 1] - level[i]
            stepi = (step[i + 1] + step[i]) / 2
            yi_pos = sigmoid(x - stepi, up_down=0, scale_x=scale_x, scale_y=scale_y)
            yi_neg = sigmoid(x + stepi, up_down=-1, scale_x=scale_x, scale_y=scale_y)
            y = y + yi_pos + yi_neg
        return y

    steps = [0, threshold / 1.2, threshold, threshold * 3, threshold * 6]
    levels = [0, 0.15, 0.3, 0.6, 1]
    return sigmoid_steps(x, steps, levels)
