import numpy as np
import pyfiglet

def normalize_rows(x):
    """ Row normalization function    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x ** 2, axis=1)).reshape((N, 1)) + 1e-30
    return x

def get_answer():
    answer = input("Type the answer [a-d,n]:")
    if answer in ["a", "b", "c", "d", "n"]:
        return answer
    return get_answer()


def print_ascii(text):
    ascii_banner = pyfiglet.figlet_format(text)
    print(ascii_banner)