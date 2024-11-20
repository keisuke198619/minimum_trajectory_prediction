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

## Results
| Model name / pred err    | mean pos [m]  | mean vel [m/s] | endpoint pos[m] | endpoint vel[m/s]|
| -------------------------|-------------- | -------------- | --------------- | -----------------|
| 1. Velocity              | 4.85 +/- 1.30 | 3.68 +/- 0.84  |  9.28 +/- 2.57  |  3.96 +/- 1.42   |  
| 2. RNN-Gauss (10 epochs) | 3.64 +/- 1.23 | 2.79 +/- 0.46  |  6.83 +/- 2.45  |  2.83 +/- 0.67   |


## Further details 
* https://github.com/keisuke198619/PO-MC-DHVRNN
* https://github.com/keisuke198619/C-OBSO
