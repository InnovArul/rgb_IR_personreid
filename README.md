# rgb_IR_personreid

## RGB-D ICCV-2017 paper

Author's webpage for ICCV2017 paper **"RGB-Infrared Cross-Modality Person Re-Identification"**:
[http://isee.sysu.edu.cn/project/RGBIRReID.htm](http://isee.sysu.edu.cn/project/RGBIRReID.htm)

## Dataset

Download the dataset from the following link and extract contents to the folder `datasets/sysu_mm01`.

[Dropbox (from ICCV-2017)](https://www.dropbox.com/sh/v036mg1q4yg7awb/AABhxU-FJ4X2oyq7-Ts6bgD0a?dl=0 )

The folder contents of `datasets` folder will look like:

```
./datasets
    - sysu_mm01
        -- cam1
        -- cam2
        -- cam3
        -- cam4
        -- cam5
        -- cam6
        -- exp

```

## Training

To train the model (`ResNet6` from `model.py`), go to `src_py` folder in command prompt and execute:

```
python doall.py
```

The training logs and models will be saved under `<root>/scratch` folder.

## Testing

To test the model, go to `src_py` folder and edit the file `doall_test.py` file to place the path of pretrained model:

```
pretrained_model = "../scratch/..."
```
Then execute the command below to test the model:
```
python doall_test.py
```
The features of all the images from different cameras will be stored with file names suffixed `_camX` where `X=cam number` under the corresponding `scratch` log folder. These features will be used by Matlab evaluation code to calculate the relevant metrics.

## Metrics calculation

To evaluate and get the metrics, the authors have release a Matlab evaluation script in the github repository: [https://github.com/wuancong/SYSU-MM01/blob/master/evaluation](https://github.com/wuancong/SYSU-MM01/blob/master/evaluation).

Given the features, the evaluation code in their repository calculates the rank-1, mAP metrics. Open the `demo.m` file inside `<root>/src_py/SYSU-MM01/evaluation/` directory and give the relevant information regarding feature path, result folder, prefix of the model and execute the `demo.m` file.
