import math
from collections import OrderedDict


class Signal:
    '''
    A class that represents a Signal.
    With just X values and NaN for y values, it is an uninitialized signal- eg before running generate.
    '''

    def __init__(self, x, y=None):
        '''
        :param x:
        :param y:
        '''

        if y is not None:
            try:
                test = y[0]
            except:
                raise ValueError("Y must be a list or otherwise indexable")
        try:
            test = x[0]
        except:
            raise ValueError("X must be a list or otherwise indexable")

        self._x = tuple(x)

        try:
            self._y = tuple(y)
        except TypeError:
            self._y = tuple()

        self._graph=self._init_graph()

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def q_min(self):
        return self.x[0]  # the first value of x

    @property
    def q_max(self):
        return self.x[-1]  # the last value of x

    @property
    def generated_points(self):
        return len(self.x) - 1  # len(x)-1?

    @property
    def graph(self):
        return self._graph

    def _init_graph(self):
        y = self.y
        if y is None or not len(y):
            y = ["NaN"] * len(self.x)

        assert len(y) == len(self.x)

        graph = OrderedDict()
        for i in range(len(self.x)):
            graph[self.x[i]] = y[i]

        return graph

    @classmethod
    def create_x_vector(cls, q_max=7.5, q_min=0, generated_points=800):
        '''
        create Signal instance from the received params

       :param q_max: max q value
       :param q_min: min q value
       :param generated_points: number of pointd between q min and q max
       :return: instance of Signal class with values that fit to the input params

        '''
        q_max = float(q_max)
        q_min = float(q_min)
        generated_points = int(generated_points)

        if generated_points <= 0:
            raise ValueError("generate points must be greater than zero")
        try:
            generated_points = int(generated_points)
        except:
            raise ValueError("generated points must be integer")

        if q_max <= 0:
            raise ValueError("q_max must be greater than zero")

        if q_min < 0:
            raise ValueError("q_min must be greater than zero")
        if q_min > q_max:
            raise ValueError("q_min must be smaller than q_max")

        generated_points += 1
        qvec = []
        for i in range(generated_points):
            val = q_min + (((q_max - q_min) * float(i)) / (generated_points - 1))
            qvec.append(val)
        return cls(qvec)

    @classmethod
    def load_from_unordered_dictionary(cls, udict):
        '''
           gets unordered dict of qs and their intensities and return signal instance that fit to the dict

          :param udict: unordered dictionary of qs as keys and ys as values
          :return: instance of Signal class with values that fit to the dict

           '''
        items = list(udict.items())
        return cls.load_from_unordered_pairs(items)

    @classmethod
    def load_from_unordered_pairs(cls, items):
        '''
           gets list of unordered pairs- qs and their intensities and return signal instance that fit to the list

          :param items: unordered list of pairs  [q1,y1]..
          :return: instance of Signal class with values that fit to the list

           '''
        items.sort(key=lambda item: item[0])
        x = [item[0] for item in items][:]
        y = [item[1] for item in items][:]
        return Signal(x, y)

    @classmethod
    def read_from_file(cls, filename):
        '''
             gets a file name and load the file as a Signal class

            :param filename: signal file name
            :return: instance of Signal class

             '''
        x_vec = []
        y_vec = []
        with open(filename) as signal_file:
            for line in signal_file:
                if '#' in line:  # a header line
                    continue
                values = line.split()
                if len(values) > 1:  # two float values
                    try:
                        x = float(values[0])
                        y = float(values[1])
                        x_vec.append(x)
                        y_vec.append(y)
                    except ValueError:  # in the c++ code, if they weren't floats, it just continued
                        continue
        return cls(x_vec, y_vec)

    def get_validated(self):
        '''
             returns a signal with no negative intensity values (remove xs and ys when the ys are negative)

            :return: instance of Signal class

             '''
        pos_indices = [i for i in range(len(self.y)) if self.y[i] >= 0.]
        if len(pos_indices) == len(self.y):
            return self
        print("Warning: There are " + str(len(self.y) - len(pos_indices)) + " negative intensity values. "
                                                                            "These are being removed from the signal")

        xs = []
        ys = []
        for index in pos_indices:
            xs.append(self.x[index])
            ys.append(self.y[index])

        return Signal(xs, ys)

    def apply_resolution_function(self, sigma):
        '''
             gets a sigma value and apply resolution on the y according to the sigma\
              returns signal instance with the new value

             :param sigma: sigma value
             :return: a signal instance

             '''
        x = list(self.x)
        y = list(self.y)
        length = len(y)

        num_sigma = 3.0
        twosigmasquared = 2 * sigma * sigma
        preExp = 1.0 / math.sqrt(math.pi * twosigmasquared)
        prev_beg = 0
        for i in range(length):
            while (x[i] - x[prev_beg]) > (num_sigma * sigma):
                prev_beg += 1
            totalWt = 0.0
            val = 0.0
            for j in range(prev_beg, length):
                if (x[j] - x[i]) > (num_sigma * sigma):
                    break
                neg_square = - (x[j] - x[i]) * (x[j] - x[i])
                eexp = math.exp(neg_square / twosigmasquared)
                weight = preExp * eexp
                totalWt += weight
                val += self.y[j] * weight
            y[i] = val / totalWt
        return Signal(x, y)
