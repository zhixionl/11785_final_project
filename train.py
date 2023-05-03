from utils import TrainOptions
from train import Trainer
#from train.trainer_spin import Trainer
import wandb

if __name__ == '__main__':

    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    wandb.login(key="Sorry this is a secret")
    wandb.init(project="Final Project", entity="zhixionl")

    # Create your wandb run
    run = wandb.init(
        #name = "ablation: mlp_5layers_convnext_batchnorm+AdamW", 
        name = "Convnext 2_layers",
        reinit = True,
        project = "Final Project"
    )

    trainer.train()
