
# YAML Introduction

## What is YAML?
- **YAML** stands for "YAML Ain't Markup Language".
- It's a human-readable data serialization format.
- Often used for configuration files and data exchange between languages with different data structures.

## Why use YAML?
- Simple syntax makes it easy to read and write.
- Widely supported in many programming languages.
- Perfect for configuration files in ML projects.

## Key Points:
- YAML files use indentation to define structure.
- Relies on key-value pairs, lists, and dictionaries.


### **2. Basic YAML Syntax**

# Basic YAML Syntax

## Key-Value Pairs
```yaml
key: value
```

## Lists
```yaml
items:
  - item1
  - item2
  - item3
```

## Nested Structures (Dictionaries within dictionaries)
```yaml
parent_key:
  child_key: value
```

## Comments
- Use `#` to add comments in YAML.
```yaml
# This is a comment
key: value  # This is an inline comment
```

## Important Notes:
- **Indentation matters**: Use spaces, not tabs, for indentation.
- **Data Types**: YAML automatically detects strings, numbers, booleans, and more.


### **3. Creating and Using YAML in ML Projects**

# Using YAML in ML Projects

## Example Configuration File
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

## Explanation:
- **project_name**: The name of your ML project.
- **experiment**: Contains the details of the experiment (e.g., name, type).
- **data**: Paths to your data and how to split it.
- **model**: Model type and hyperparameters for training.
- **output**: Where to save the model and logs.

## Using YAML in Python
- **Loading YAML**:
```python
import yaml

# Load the YAML configuration file
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Access configuration values
print(config['experiment']['name'])  # Outputs: baseline_model
```

- **Accessing Values**:
  - **config['key']**: Access top-level keys.
  - **config['parent_key']['child_key']**: Access nested keys.


### **4. Hands-On: Writing Your Own YAML File**

# Hands-On: Writing Your Own YAML File

## Steps:
1. **Create a new file** called `config.yaml`.
2. **Write the following configuration**:
```yaml
project_name: "my_first_project"

data:
  source: "data/sample.csv"
  validation_split: 0.3

model:
  type: "decision_tree"
  hyperparameters:
    max_depth: 4

output:
  model_path: "output/models/"
```

3. **Load this YAML file in Python**:
```python
import yaml

with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

print(config['project_name'])  # Outputs: my_first_project
```

## Experiment:
- **Change a value** in the YAML file (e.g., `max_depth`) and observe how it affects your model configuration.


### **5. Common Pitfalls and Best Practices**

# Common Pitfalls and Best Practices

## Pitfalls:
- **Incorrect Indentation**: YAML is sensitive to spaces. Always use consistent indentation.
- **Using Tabs Instead of Spaces**: Always use spaces for indentation, as YAML does not support tabs.

## Best Practices:
- **Keep It Simple**: Avoid overly complex nested structures unless necessary.
- **Document Your YAML**: Use comments to explain sections of your YAML file.
- **Version Control**: Track your YAML files with version control (e.g., Git) to manage changes.

## Final Tips:
- **Use a Linter**: Some code editors have YAML linters to help detect syntax issues.
- **Validate YAML**: Use online YAML validators to check your syntax before running it in your project.
