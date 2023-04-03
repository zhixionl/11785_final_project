# Multi-View 3D Human Pose and Shape Estimation via Model-fitting in the Loop (Multi-SPIN)

Our code is based on the paper: 
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
Currently we had problems with pyrender, but we will release the instructions for installation in the future

## Fetch data
We provide a script to fetch the necessary data for training and evaluation. You need to run:
```
./fetch_data.sh
```
The GMM prior is trained and provided by the original [SMPLify work](http://smplify.is.tue.mpg.de/), while the implementation of the GMM prior function follows the [SMPLify-X work](https://github.com/vchoutas/smplify-x). Please respect the license of the respective works.

Besides these files, you also need to download the *SMPL* model. You will need the [neutral model](http://smplify.is.tue.mpg.de) for training and running the demo code, while the [male and female models](http://smpl.is.tue.mpg.de) will be necessary for evaluation on the 3DPW dataset. Please go to the websites for the corresponding projects and register to get access to the downloads section. In case you need to convert the models to be compatible with python3, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

## Run demo code
*Training* To train our network, we would need both json files from Openpose 2D keypoints detection and images files (We strongly recommend to use ffmpeg instead of opencv-python as suggested in the original gihtub code as it could save time and space a lot. The training json files were provided after fetching data, but you would need to include your own json files when evaluate the model on your own video files. More details will be released in the future. 

Training from scratches (what we submitted for the midterm report
```
python3 train.py --name train_example --run_simplify --num_epoch=10 --summary_steps=300 --checkpoint_steps=300
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
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```

You can also save the results (predicted SMPL parameters, camera and 3D pose) in a .npz file using ```--result=out.npz```.

## Run training code
Due to license limitiations, we cannot provide the SMPL parameters for Human3.6M (recovered using [MoSh](http://mosh.is.tue.mpg.de)). Even if you do not have access to these parameters, you can still use our training code using data from the other datasets. Again, make sure that you follow the [details for data preprocessing](datasets/preprocess/README.md).
