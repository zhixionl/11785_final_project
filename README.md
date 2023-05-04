# Multi-view 3D Human Body Mesh Reconstruction through Iterative Regression and Optimization (Multi-SPIN)

#### The names are ordered in alphabetical order but everyone in the group all contributed equally to this project. 
**Jack Zhixiong Li**,   **Samuel Yu-Cheng Lin**,   **Felicia Zhixin Luo**,   **Boyi Qian**

Our code was implemented based on the paper to convert it as a multi-view method: 
**Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop**  
[Nikos Kolotouros](https://www.seas.upenn.edu/~nkolot/)\*, [Georgios Pavlakos](https://www.seas.upenn.edu/~pavlakos/)\*, [Michael J. Black](https://ps.is.mpg.de/~black), [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)  
ICCV 2019  
[[paper](https://arxiv.org/pdf/1909.12828.pdf)] [[project page](https://www.seas.upenn.edu/~nkolot/projects/spin/)]

## Installation instructions
```
conda create -n 11785_project python=3.8
conda activate 11785_project
pip install -U pip
pip install -r requirements.txt
```

Currently we had problems with neural render , but we will release the instructions for installation in the future
* We have met many issues with the library dependencies, and here is the document for the potential issues and corresponding sotlutions: 
* https://docs.google.com/document/d/1JF2b6cpeUDmMMjE2h9JngXvWX_GkkB01StVps6XqJpI/edit?usp=sharing
* We have tested the code on torch = 2.0.0, torchvision = 0.15.1, and cuda = 11.7 on Ubuntu 20.04. Please make sure your library dependencies are correct
* Instead of the neural render, we used the pytorch3d with customized algorithms to display our vertices results:

<p float="left">
   <img src="https://github.com/zhixionl/11785_final_project/blob/main/assets/video_gif%201.gif" width="100%">
</p>


## Fetch data
We provide a script to fetch the necessary data for training and evaluation. You need to run:
```
./fetch_data.sh
```
You also need to download the *SMPL* model. You will need the [neutral model](http://smplify.is.tue.mpg.de) for training and running the demo code, while the [male and female models](http://smpl.is.tue.mpg.de) will be necessary for evaluation on the 3DPW dataset. Please go to the websites for the corresponding projects and register to get access to the downloads section. In case you need to convert the models to be compatible with python3, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

## Run demo code
*Training* To train our network, we would need both json files from Openpose 2D keypoints detection and images files (We strongly recommend to use ffmpeg instead of opencv-python as suggested in the original gihtub code as it could save time and space a lot. The training json files were provided after fetching data, but you would need to include your own json files when evaluate the model on your own video files. More details will be released in the future. 

Training from scratches (what we submitted for the midterm report)
```
python3 train.py --name train_example --run_smplify --num_epoch=10 --summary_steps=300 --checkpoint_steps=300
```
If you need to resume the training, then simply add the args: --checkpoints ==...
The checkpoint will be saved in logs with default settings. 

## Run evaluation code
Besides the demo code, we also provide code to evaluate our models on the datasets we employ for our empirical evaluation. Before continuing, please make sure that you follow the [details for data preprocessing](datasets/preprocess/README.md).

Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=mpi-inf-3dhp --log_freq=5
```

Running the above command will compute the MPJPE and Reconstruction Error on the MPI-INF-3DHP dataset. The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```

You can also save the results (predicted SMPL parameters, camera and 3D pose) in a .npz file using ```--result=out.npz```. Since we don't have the access to Human3.6M for now, as explained below, we only use MPI-INF-3DHP for current implementation for the midterm report, but we will use NeuralAnnot for to obtain the ground truth SMPL from Human3.6M for the final project. 

## Run training code (The following note was taken from SPIN project page): 
Due to license limitiations, we cannot provide the SMPL parameters for Human3.6M (recovered using [MoSh](http://mosh.is.tue.mpg.de)). Even if you do not have access to these parameters, you can still use our training code using data from the other datasets. Again, make sure that you follow the [details for data preprocessing](datasets/preprocess/README.md).
