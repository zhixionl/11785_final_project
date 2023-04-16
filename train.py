from utils import TrainOptions
from train import Trainer
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    options = TrainOptions().parse_args()
    print("------------------------------------------INITIALIZING------------------------------------------")
    trainer = Trainer(options)
    print("------------------------------------------INIT COMPLETE------------------------------------------")
    trainer.train()
