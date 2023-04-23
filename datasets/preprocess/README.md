## Data preparation
Besides the demo code, we also provide training and evaluation code for our approach. To use this functionality, you need to download the relevant datasets.
The datasets that our code supports are:
1. [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
2. [Human3.6M](http://vision.imar.ro/human3.6m/description.php)

More specifically:
1. **MPI-INF-3DHP**: Again, we use this dataset for training and evaluation. You need to download the data from the [dataset website](http://gvv.mpi-inf.mpg.de/3dhp-dataset). The expected fodler structure at the end of the processing looks like:
```
${MPI_INF_3DHP root}
|-- mpi_inf_3dhp_test_set
    |-- TS1
|-- S1
    |-- Seq1
        |-- imageFrames
            |-- video_0
```
Then, you need to edit the ```config.py``` file with the ```${MPI_INF_3DHP root}``` path.

Due to the large size of this dataset we subsample the frames used by a factor of 10. Also, in the training .npz files, we have included fits produced by SMPL fitting on 2D joints predictions from multiple views. Since, the quality of these fits is not perfect, we only keep 60% of the fits fixed, while the rest are updated within the typical SPIN loop.

2. **Human3.6M**: Unfortunately, due to license limitations, we are not allowed to redistribute the MoShed data that we used for training. We only provide code to evaluate our approach on this benchmark. To download the relevant data, please visit the [website of the dataset](http://vision.imar.ro/human3.6m/description.php) and download the Videos, BBoxes MAT (under Segments) and 3D Positions Mono (under Poses) for Subjects S9 and S11. After downloading and uncompress the data, store them in the folder ```${Human3.6M root}```. The sructure of the data should look like this:
```
${Human3.6M root}
|-- S9
    |-- Videos
    |-- Segments
    |-- Bboxes
|-- S11
    |-- Videos
    |-- Segments
    |-- Bboxes
```
You also need to edit the file ```config.py``` to reflect the path ```${Human3.6M root}``` you used to store the data. 

```

### Generate dataset files
After preparing the data, we continue with the preprocessing to produce the data/annotations for each dataset in the expected format. With the exception of Human3.6M, we already provide these files and you can get them by running the ```fetch_data.sh``` script. If you want to generate the files yourself, you need to run the file ```preprocess_datasets.py``` from the main folder of this repo that will do all this automatically. Keep in mind that this assumes you have already run OpenPose on the images of all datasets used for training and you have provided the respective folders with the OpenPose ```.json``` files in the ```config.py``` folder. Depending on whether you want to do evaluation or/and training, we provide two modes:

If you want to generate the files such that you can evaluate our pretrained models, you need to run:
```
python preprocess_datasets.py --eval_files
```
If you want to generate the files such that you can train using the supported datasets, you need to run:
```
python preprocess_datasets.py --train_files
```
