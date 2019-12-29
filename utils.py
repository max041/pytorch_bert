import pickle


def save_pickle(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]