import random
import json
import subprocess

while True:
    modelConf = {
        "convo_1": random.choice([False, True]),
        "convo_2": random.choice([False, True]),
        "convo_3": random.choice([False, True]),
        "convo_4": random.choice([False, True]),
        "convo_5": random.choice([False, True]),
        "convo_depth_factor": random.choice([1, 2, 3, 4, 5]),
        "add_pooling": True, # random.choice([False, True]),
        "dropout_1": True, # random.choice([False, True]),
        "dropout_2": False, # random.choice([False, True]),
        "dense_3": True, # random.choice([False, True]),
        "dropout_3": True, # random.choice([False, True]),
        "dense_1_factor": random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "dense_2_factor": random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "dense_3_factor": random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "example_imbalance": random.choice([2, 3, 4, 5])
    }
    
    conv_layers = 0
    for key in ["convo_1", "convo_2", "convo_3", "convo_4", "convo_5"]:
        if modelConf[key]:
            conv_layers += 1
    if conv_layers < 3:
        continue

    json.dump(modelConf, open("modelConf.json", "w"))
    subprocess.call("python train.py", shell=True)

