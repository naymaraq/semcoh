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


def get_float_answer():
    answer = input("Enter the score [0-10,n]:")
    try:
        if answer == 'n': return answer
        answer = float(answer)
        if 0 <= answer <= 10:
            return answer
        else:
            print("Score must be in range [0,10]")

    except Exception as e:
        print(e)
    return get_float_answer()


def print_ascii(text):
    ascii_banner = pyfiglet.figlet_format(text)
    print(ascii_banner)
