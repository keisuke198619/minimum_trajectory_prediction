## Minimum Trajectory Prediction Code
This is the python code for minimum trajectory prediction using STATSPerform trajectory data (should not be shared)

## Author
Keisuke Fujii 

## Requirements

* To install requirements:

```setup
pip install -r requirements.txt
```

## Usage
 
* Run `run.sh` for a simple demonstration of training and test using the STATSPerform dataset.

* Actual commands in training and test of our model are also in `run.sh`.  

## Results
| Model name              |   mean pos    |   mean vel    |    max pos     |  max  vel      |
| ------------------------|-------------- | ------------- | -------------- | -------------- |
| 1. Velocity             |               |               |                |                |  
| 2. RNN-Gauss (4 epochs) | 3.32 +/- 1.45 | 2.70 +/- 0.55 |  6.16 +/- 2.88 |  3.95 +/- 0.93 |


## Further details 
* https://github.com/keisuke198619/PO-MC-DHVRNN
* https://github.com/keisuke198619/C-OBSO
