Pulsar detection with CNN
====

This project is a pulsar detection program based on the object detection method [FCOS](https://github.com/tianzhi0549/FCOS).
The program can 
1. directly read filterbank files (1-bit or others), conver it to image with downsample and normalization, 
2. input the image into the deep neural network
3. output the location (time and frequency) where pulses appear and cut the 
data patch from the whole fil as pulsar candidates. 

The process from filterbank data to candidate images is end-to-end.

The model can runing on CPU or GPU. The detection speed can be impressive with GPU.
  

The model are trained on fake pulsar data, which are generated by adding fake pulses to the observation data without pulse signals.
Details can be find in [fake_from_real.py](./fake_from_real_v2.py)


### Requirements
        python >= 3.6
        pytorch >= 1.1.0
        torchvision >= 0.4.0
        opencv-python
        sigpyproc
[Sigpyproc](https://github.com/ewanbarr/sigpyproc) is used for read filterbank data.
### Usage
Download the trained model parameter from [here](https://drive.google.com/file/d/1UQetP-7PpQPg2GvM_qBeLEmonG5e8DK-/view?usp=sharing), this model are trained on 5000 fake pulsar images
 with different signal-noise and random DM(20-2000). The train images are all from 1-bit data, so re-train may be necessary 
 when processing other kinds of data, and codes also should be modified.
```bash
python inference.py {fil path to be detected} --output {output path}
```
The program will generate a json file to save all possible pulses and 
crop every pulse as single picture for visualization.

An example filterbank data can be [download](https://drive.google.com/file/d/1NwDWzNfABNNXWi9MoMXKLCSCfX5vEMbz/view?usp=sharing) for test.


### Train
1. Create fake pulsar data as train dataset: Download the [real data](https://drive.google.com/file/d/1h7zbuIxdGN7-rlxVA6cW5oQ6zK2GC4yy/view?usp=sharing) that has been confirmed there is no pulsar in them, decompress it to 
to get the folder "./nopulse_fils". Run
    ```bash
    python fake_from_real.py
    ```
    Three folders will be created, "fake_fils", "fake_images" and "annotations" to save filterbank data, images and annotations info. The last two 
    will be used for training.
2. Training, Run
    ```bash
    python train.py configs/default.py
    ```
   Training parameters (such as training set, batch size, learning rate) has been set in "configs/default.py".
