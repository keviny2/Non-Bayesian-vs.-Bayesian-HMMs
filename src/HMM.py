import numpy as np
from Distribution import normal_pdf

class HMM():

    def __init__(self):
        self.A
        self.B
        self.init
        self.obser

    # TODO: (KEVIN) honestly this class is totally wrong, don't think i need

    def eexp(self, x):
        """

        :param x: x
        :return: exp(x)
        """
        if x == None:
            return 0
        else:
            return np.exp(x)


    def eln(self, x):
        """

        :param x: x
        :return: ln(x)
        """
        try:
            if x == 0:
                return None
            elif x > 0:
                return np.log(x)
        except ValueError:
            print("negative input error")
            raise ValueError


    def elnsum(self, x, y):  # ln(x+y)
        if x == None or y == None:
            if x == None:
                return y
            else:
                return x

        else:
            if x > y:
                return x + self.eln(1 + np.exp(y - x))
            else:
                return y + self.eln(1 + np.exp(x - y))


    def elnproduct(self, x, y):  # ln(x) + ln(y)
        if x == None or y == None:
            return None
        else:
            return x + y

    def viterbi_robust(self, data, initial, A, B):
        nrow = len(data)

        path = np.zeros(nrow, dtype=int)
        prob = np.zeros((nrow, self.num_states))
        state = np.zeros((nrow, self.num_states))

        for i in range(self.num_states):
            prob[0, i] = self.elnproduct(self.eln(initial[i]), self.eln(normal_pdf(data[0], B[i, 0], B[i, 1])))
        state[0] = 0

        for i in range(1, nrow):
            for j in range(self.num_states):
                list = np.zeros(6)
                for k in range(self.num_states):
                    list[k] = self.elnproduct(prob[i - 1, k], self.elnproduct(self.eln(A[k, j]),
                                                                              self.eln(normal_pdf(data[i], B[j, 0],
                                                                                                  B[j, 1]))))

                prob[i, j] = np.max(list)
                state[i, j] = np.argmax(list)

        path[nrow - 1] = np.argmax(prob[nrow - 1])
        for i in range(nrow - 2, -1, -1):
            path[i] = state[i + 1, path[i + 1]]

        return path, prob, state


