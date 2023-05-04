import os
import pickle


def save_pr(path, file, data, names):
    """
    Save the data as a dict in a file located in the path
    :param path: the location of the saved file
    :param file: the name of the file
    :param data: the data to be saved
    :param names: the names of the data
    Notes: 1. Conventionally, use the suffix \'.pr\'. 2. If the folder does not exist, system will
    automatically create one. 3. use \'load_pr\' to load a .pr file
    Example:
    >>> x = 1
    >>> y = 'good'
    >>> save_pr('.\\test', 'ok.pr', [x, y], ['name1', 'name2'])
      You have a file '.\\test\\ok.pr'
    >>> z = load_pr('.\\test\\ok.pr')
      z = {'name1': 1, 'name2': 'good'}
      type(z) is dict
    """
    mkdir(path)
    # print(os.path.join(path, file))
    s = open(os.path.join(path, file), 'wb')
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    pickle.dump(tmp, s)
    s.close()
    
    
def load_pr(path_file, names=None):
    """
    Load the file saved by save_pr as a dict from path
    :param path_file: the path and name of the file
    :param names: the specific names of the data you want to load
    :return  the file you loaded
    Notes: the file you load should be a  \'.pr\' file.
    Example:
        >>> x = 1
        >>> y = 'good'
        >>> z = [1, 2, 3]
        >>> save_pr('.\\test', 'ok.pr', [x, y, z], ['name1', 'name2', 'name3'])
        >>> A = load_pr('.\\test\\ok.pr')
          A = {'name1': 1, 'name2': 'good'}
        >>> y, z = load_pr('\\test\\ok.pr', ['y', 'z'])
          y = 'good'
          z = [1, 2, 3]
    """
    if os.path.isfile(path_file):
        s = open(path_file, 'rb')
        if names is None:
            data = pickle.load(s)
            s.close()
            return data
        else:
            tmp = pickle.load(s)
            if type(names) is str:
                data = tmp[names]
                s.close()
                return data
            elif type(names) is list or type(names) is tuple:
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                s.close()
                return tuple(data)
    else:
        return False


def mkdir(path):
    """
       Create a folder at your path
       :param path: the path of the folder you wish to create
       :return: the path of folder being created
       Notes: if the folder already exist, it will not create a new one.
    """
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag

