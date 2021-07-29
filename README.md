# DeepDance2

### Environment
* **Tested OS:** MacOS, Linux
* Python >= 3.6
### How to install
1. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Install MuJoCo following the steps [here](https://github.com/openai/mujoco-py#install-mujoco). Note that [mujoco-py](https://github.com/openai/mujoco-py) (MuJoCo's python binding) is already installed in step 1. This step is to install the actual MuJoCo library. You will need to apply for a [MuJoCo Personal License](https://www.roboti.us/license.html) (free for students).
3. Set the following environment variable to improve multi-threaded sampling performance:    
    ```
    export OMP_NUM_THREADS=1
    ```
The current model is only able to repeatly learn a singel clip of version, but we want it to generate model accroding to the input musice. the framework of the model:



the framework of our expected model:

