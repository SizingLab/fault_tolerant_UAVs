Install
-------
You can install the required dependencies by opening a command prompt in the unziped folder and run the following command:
``` {.bash}
$ pip install -r requirements.txt
```


Usage
-------
The *fault-tolerant-sizing.py* file contains Python functions to assess the controllability 
and achieve fault-tolerant sizing of fixed-wing and hybrid FW-VTOL UAVs.

A linearized model of a UAV is provided in the `main` function. 
In the same function, the failure matrix defines the failure case to assess. 
To run the controllability assessment and obtain the sizing factors for the defined failure case, run the `main` function.
Finally, repeat the operation for all failure cases to assess.
