Pedestrian Detection Dataset Generation in CARLA 
=============================================



Extraction of CARLA Data
---------------------------
This API helps you information about the RGB Image, the Semantic Segmentation Image and the Bounding Box of Objects from the CARLA Server. In addition, it allows users to capture these informations at regular intervals and keyboard inputs.

* CARLA Server on Linux
> ./CarlaUE4.sh

* CARLA Server on Windows
> CarlaUE4.exe

* To check if CARLA is running, open another Command Prompt and run:
> netstat -ano | findstr 2000

Dataset Generation
---------------------------

### Dataset Structure

The dataset is organized hierarchically:
```
carla_dataset/
├── Town01/
│   ├── monk_group_1/
│   │   ├── cloud_0_rain_0_fog_0/
│   │   │   ├── RGB/
│   │   │   │   ├── 0.png
│   │   │   │   ├── ...
│   │   │   ├── ... (other data folders)
│   │   ├── ... (other weather conditions)
│   ├── ... (other monk groups)
├── Town03/
│   ├── ... (similar structure)
├── Town05/
│   ├── ... (similar structure)
```

### Dataset Variables

The dataset generation covers the following variables:

1. **Towns**: Three CARLA environments
   - Town01
   - Town03
   - Town05

2. **Monk Skin Tone Groups**: 10 different pedestrian appearance groups
   - Each group represents a specific pedestrian blueprint in CARLA

3. **Weather Conditions**: 13 different conditions with variations in:
   - Cloudiness (0%, 25%, 50%, 75%, 100%)
   - Rain (0%, 25%, 50%, 75%, 100%)
   - Fog (0%, 25%, 50%, 75%, 100%)

### Running the Dataset Generation

To generate the full dataset:

1. Ensure CARLA server is running
2. Execute the dataset generation script:
   ```bash
   bash main_dataset.sh
   ```

The script will:
- Process each town sequentially
- For each town, iterate through each monk skin tone group
- Apply all weather conditions for each group
- Collect approximately 200 images per configuration
- Skip configurations that already have enough images

### Script Execution Details

The script performs the following steps for each configuration:

1. Creates the required directory structure
2. Spawns vehicles and specific pedestrian types 
3. Sets the weather conditions
4. Captures images and relevant data
5. Monitors progress and retries if needed (up to 50 attempts per configuration)

### Requirements

- CARLA simulator
- Python 3.x with CARLA client library
- Required Python scripts:
  - spawn.py - For spawning actors in the simulation
  - weather.py - For setting weather conditions
  - self_extract.py - For capturing and saving data

### Expected Output

For each configuration, the script generates:
- RGB images in PNG format
- Additional data (depending on self_extract.py configuration)

The target is to collect 200 images per configuration, resulting in a comprehensive dataset covering various environmental conditions and pedestrian appearances.

