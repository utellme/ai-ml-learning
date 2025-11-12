# Task: Create a YAML Configuration File for Your ML Project

## Objective
In this task, you will create a YAML configuration file that will serve as the central configuration point for your machine learning project. The YAML file will define various aspects of the project, including the experiment details, data sources, model hyperparameters, and output paths.

## Steps to Complete the Task

### 1. Understanding the Structure
Before you begin, let's break down the structure of the YAML file you'll be creating. This configuration file will contain the following sections:

- **project_name**: The name of your machine learning project.
- **experiment**: Information about the experiment, including its name and type.
- **data**: Details about the data source and how it should be split into training, validation, and test sets.
- **model**: The type of model you'll be using and its hyperparameters.
- **output**: Paths where your model and logs will be saved.

### 2. Create a New YAML File

1. **Create a new file** in your project directory and name it `config.yaml`.

2. **Open the file** in your text editor.

### 3. Define the Project Name
Start by specifying the name of your project:

```yaml
project_name: "my_ml_project"
```

### 4. Define the Experiment Details
Next, add a section for your experiment details. This includes the name of the experiment and its type (e.g., classification, regression):

```yaml
experiment:
  name: "baseline_model"
  type: "classification"
```

### 5. Specify the Data Configuration
Now, define the data section. Here, you'll specify the path to your dataset and how you want to split the data:

```yaml
data:
  source: "data/iris.csv"
  validation_split: 0.2
  test_split: 0.1
```

- **source**: The path to your dataset file.
- **validation_split**: The portion of the dataset to be used for validation.
- **test_split**: The portion of the dataset to be used for testing.

### 6. Set Up the Model Configuration
Then, define the model you will use, along with its hyperparameters:

```yaml
model:
  type: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 5
    random_state: 42
```

- **type**: The type of model (e.g., random_forest).
- **hyperparameters**: The hyperparameters for the model, such as the number of estimators, maximum depth, and random state.

### 7. Define the Output Paths
Finally, specify where you want to save the trained model and any logs generated during training:

```yaml
output:
  model_path: "models/"
  logs_path: "logs/"
```

- **model_path**: The directory where the model will be saved.
- **logs_path**: The directory where logs will be stored.

### 8. Review and Save
After completing the steps above, your `config.yaml` file should look like this:

```yaml
project_name: "my_ml_project"

experiment:
  name: "baseline_model"
  type: "classification"

data:
  source: "data/iris.csv"
  validation_split: 0.2
  test_split: 0.1

model:
  type: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 5
    random_state: 42

output:
  model_path: "models/"
  logs_path: "logs/"
```

- **Save the file** after verifying that everything is correct.
