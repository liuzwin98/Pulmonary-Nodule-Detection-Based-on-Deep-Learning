
# Multi-scale Gradual Integration CNN for False Positive Reduction in Pulmonary Nodule Detection

<font size=7 color=red>Please see https://github.com/ku-milab/MGICNN for details.</font> 

(Paper Here: https://doi.org/10.1016/j.neunet.2019.03.003)


## Usage

For training:

`python main.py --data_path=PATH --summ_path_root=PATH --fold=0 --maxfold=5 --multistream_mode=0 --model_mode=0 --train`

For testing:

`python main.py --data_path=PATH --summ_path_root=PATH --fold=0 --maxfold=5 --multistream_mode=0 --model_mode=0 --test --tst_model_path=PATH --tst_epoch=40`

* Specify your data path (--data_path) and path to save your results and summary (--summ_path_root). Unzipped LUNA16 dataset should be inside "(--data_path)/raw/" folder.
```
Example
--data_path=/home/jsyoon/MGICNN/dataset/
/home/jsyoon/MGICNN/dataset/raw/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd
/home/jsyoon/MGICNN/dataset/raw/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.raw
...
/home/jsyoon/MGICNN/dataset/raw/candidates_V2.csv
```
* Specify fold to train (--fold) and maximum number of folds (--maxfold).
* Specify which multistream mode to use (--multistream_mode). (0-element(proposed), 1- concat, 2-1x1 comv)
* Specify which model to use (--model_mode). (0-proposed, 1-RI , 2-LR, 3-ZI, 4- ZO)
* Specify train or test (--train or --test and --tst_model_path/--tst_epoch).


