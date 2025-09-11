import torch, pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_meta = unpickle('./datasets/cifar-10-batches-py/batches.meta')

if __name__ == "__main__":
    ...