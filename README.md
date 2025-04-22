# ROB_530_Vision_Based_Gyroscope

This project uses only visual data to create and estimate 6DoF pose estimation, creating a vision-based gyroscope. A breakdown of the project is in the paper. 
Authors: Niva Ranavat, Kunal Atram, Shuhan Guo, Zhiyi Wang


## Project Setup Instructions

Follow the steps below to set up the project using a Python virtual environment.

---

### Step 1: Clone the Repository

```bash
git clone ROB_530_Vision_Based_Gyroscope
cd ROB_530_Vision_Based_Gyroscope
```


### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment
On macOS/Linux:

```bash
source venv/bin/activate
```


On Windows:

```bash
venv\Scripts\activate
```


### Step 4: Install the Requirements

```bash
pip install -r requirements.txt
```

## Step 5: Download Required Datasets (TUM RGB-D)

This project requires two TUM RGB-D datasets:

- `rgbd_dataset_freiburg1_xyz`
- `rgbd_dataset_freiburg1_rpy`

### Instructions

**Step 1:** Go to the official TUM RGB-D dataset website:  
[https://vision.in.tum.de/data/datasets/rgbd-dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset)

**Step 2:** Scroll down to the **freiburg1** section and download the following:

- [`rgbd_dataset_freiburg1_xyz.tgz`](https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz)
- [`rgbd_dataset_freiburg1_rpy.tgz`](https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.tgz)

**Step 3:** Extract the downloaded `.tgz` files.

**Step 4:** Create a folder in your project called `data` and move the extracted folders into it.  
  1. **Navigate to the project folder**  
   Open a terminal and `cd` into the project root directory.

  2. **Create a `data` directory**  
   This is where the datasets will be stored.

   ```bash
   mkdir -p data
   cd data
   ```




### Step 6: Run the Project

```bash
python main.py
```
