import numpy as np
import time


def pt_3d_format(pts_3d):
    pass


def pts_2d_format(pts_2d):
    pass


def pts_2d_2_3d_homo(pts_2d):
    assert pts_2d.shape[1] == 2
    return np.concatenate([np.copy(pts_2d), np.ones((pts_2d.shape[0], 1))], axis=-1)


class timer:
    def __init__(self):
        self.time_accounting = {}

    def start(self, name=''):
        name = str(name)
        self.time_accounting[name] = time.time()

    def end(self, name=''):
        name = str(name)
        self.time_accounting[name] = time.time() - self.time_accounting[name]

    def __str__(self):
        spacing = '     '
        output = 'time accounting:\n'
        for name in self.time_accounting.keys():
            output += spacing + str(name) + ' : ' + str(self.time_accounting[name]) + '\n'
        return output


def test_timer():
    t = timer()

    t.start(3)
    t.start(1)
    for i in range(1000000): pass
    t.end(1)

    t.start(2)
    for i in range(1000000): pass
    t.end(3)

    for i in range(10000):
        pass
    t.end(2)

    print(t)


def main():
    test_timer()


if __name__ == '__main__':
    main()
