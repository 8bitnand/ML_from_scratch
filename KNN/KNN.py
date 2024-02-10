import os 
from typing import List
import matplotlib.pyplot as plt
 
def load_data(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(line.split(","))

    return data
    

def process_data(data:List[List[str]]):
    data_processed = []
    labels = set()
    for item in data:
            data_processed.append(
                [float(item[0]), float(item[1]), float(item[2]), float(item[3]), item[4].strip()]
            )
            labels.add(item[4].strip())
    
    return data_processed, list(labels)


def save_plot(data_processed, labels):

    x = [ item[0] for item in data_processed ]
    y = [ item[1] for item in data_processed ]
    color = ["r","g", "b","y"]
    color_map = {l:c for l,c in zip(labels, color)}
    c = [color_map[item[-1]] for item in data_processed]
    plt.scatter(x, y, color=c)
    plt.savefig("KNN/iris_0_1.png")

def train_test_split(data_processed, labels):
    
    data_on_labels = { label:[] for label in labels}

    for item in data_processed:
        data_on_labels[item[-1]].append(item[:-1])

    train_test_data = { label: [] for label in labels }
    for key in labels:
        train_test_data[key].append({
            'train': data_on_labels[key][:-10],
            'test':data_on_labels[key][10:]
        })

    return train_test_data

if __name__ == "__main__":

    data = load_data("KNN/iris.csv")
    data_processed, labels = process_data(data)
    ttd = train_test_split(data_processed, labels)
    print(ttd)
    # save_plot(data_processed, labels)