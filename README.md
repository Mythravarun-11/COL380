## Preprocessing
Before compiling and running the assignment, ensure you've completed the preprocessing step:
```bash
module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load pythonpackages/3.6.0/opencv/3.4.1/gpu
python3 preprocessing.py
```

## Compilation Instructions


### Subtask1:

To compile and run subtask1:

```bash
module load compiler/gcc/7.4.0/compilervars
module load compiler/cuda/10.2/compilervars
nvcc src/assignment2_subtask1.cu -o subtask1
```

### Subtask2:

To compile and run subtask2:

```bash
module load compiler/gcc/7.4.0/compilervars
module load compiler/cuda/10.2/compilervars
nvcc src/assignment2_subtask2.cu -o subtask2
```

### Subtask3:

To compile and run subtask3:

```bash
module load compiler/gcc/7.4.0/compilervars
module load compiler/cuda/10.2/compilervars
nvcc src/assignment2_subtask3.cu -o subtask3
./subtask3
```

### Subtask4:

To compile and run subtask4:

```bash
module load compiler/gcc/7.4.0/compilervars
module load compiler/cuda/10.2/compilervars
nvcc src/assignment2_subtask4.cu -o subtask4
```

Run subtask4 with streams (1) or without streams (0):

#### With streams
```bash
./subtask4 1
```

#### Without streams

```bash
./subtask4 0
```


Problem statement is available here : https://www.cse.iitd.ac.in/~rijurekha/col380_2024/cuda_assignment.html
