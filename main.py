
import time 
import models 
import argparse 
from datetime import datetime as dt 


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to configuration file")
    ap.add_argument("-d", "--dataset", default="timit", type=str, help="Name of dataset finetuned on")
    ap.add_argument("-t", "--task", default="train", type=str, help="Task to perform. Choose between ['train', 'test']")
    ap.add_argument("-o", "--output", default=dt.now().strftime("%d-%m-%Y-%H-%M"), type=str, help="Output directory path")
    ap.add_argument("-l", "--load", default=None, type=str, help="Path to directory containing checkpoint as best_model.pt")
    args = vars(ap.parse_args())

    trainer = models.Trainer(args)
    
    if args["task"] == "train":
        trainer.train()
    
    elif args["task"] == "test":
        assert args["load"] is not None, "Please provide a checkpoint to load using --load to check test performance"
        trainer.get_test_performance()

    else:
        raise ValueError(f"Unrecognized argument passed to --task: {args['task']}")