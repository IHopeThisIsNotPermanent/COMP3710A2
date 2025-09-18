# UQ Rangpur SLURM HOW TO

#### Setup 
The guide provided by the course to access the server is pretty straight forward.

#### Running Models
The file system that the model scripts access is the one that you ssh into, so if you want to access python environments, or save out model weights, do so as you usually would. Note that the root is wherever you ran the sbatch from.

#### Running the provided example
You will need to set up a python environment as pointed to in the slurm.sh file, then navigate to the directory with the slurm.sh script in it, and run:
```bash
sbatch slurm.sh
```