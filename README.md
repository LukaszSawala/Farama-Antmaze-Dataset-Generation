# AntMaze Dataset Generation Scripts for Minari

**This code is an adaptation of the original, simple code working for AntMaze-v4 from the [Farama Foundation's minari-dataset-generation-scripts](https://github.com/Farama-Foundation/minari-dataset-generation-scripts).**

It has been significantly updated to support the **Gymnasium Robotics AntMaze-v5** environment, modern Minari standards, and robust multi-quality data generation.

---

## 📂 File Structure

* **`multiagent_create_antmaze_dataset.py`**: The primary script for generating multi-quality datasets (Expert, Medium, Beginner, Random) in one go.
* **`create_antmaze_dataset.py`**: The legacy script for generating a single dataset using a specific policy file.
* **`check_maze_dataset.py`**: Utility script for validating dataset integrity and calculating success metrics.
* **`train_ant.py`**: Script for running the training of the ant model.
*  **`reach_goal_ant.py`**: Model of the ant. 
    

---

## 1. Multi-Agent Generation (Recommended)

Use `multiagent_create_antmaze_dataset.py` to generate a complete suite of datasets with varying agent proficiencies. This script was used to generate the datasets available at [INSERT-LINK]

### Usage

Run the script and specify the desired agent quality. The script automatically handles policy loading and step counts based on the quality level.

```bash
# Generate Expert Data (100% steps, fully trained agent)
python multiagent_create_antmaze_dataset.py --quality expert

# Generate Medium Data (50% steps, partially trained agent)
python multiagent_create_antmaze_dataset.py --quality medium

# Generate Beginner Data (50% steps, early training agent)
python multiagent_create_antmaze_dataset.py --quality beginner

# Generate Random Data (25% steps, random actions)
python multiagent_create_antmaze_dataset.py --quality random
