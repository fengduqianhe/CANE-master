
from code.prepareData import prepare
from code.train import start_train

if __name__ == "__main__":
    dataset = "zhihu"
    ratio = 0.55
    #prepare(dataset, ratio)
    start_train(dataset)
