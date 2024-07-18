# Decision Tree Classifier on Car Data

This repository contains a Python script for training and evaluating a Decision Tree Classifier on a car dataset. The script demonstrates data preprocessing, training with both entropy and gini criteria, and visualization of the decision trees.

## Dataset

The dataset used in this project is `car_data.csv`. The dataset is expected to have the following columns:

- `User ID`: Unique identifier for each user (will be dropped during preprocessing)

- `Gender`: Gender of the user (`Male` or `Female`)

- `Age`: Age of the user

- `AnnualSalary`: Annual salary of the user

- `Purchased`: Whether the user purchased the car (`0` or `1`)

## Dependencies

To run this project, you need the following Python libraries:

- numpy

- pandas

- matplotlib

- scikit-learn

You can install the dependencies using pip:

```sh
pip  install  numpy  pandas  matplotlib  scikit-learn
```

## Running the Script

Make sure you have the dataset (car_data.csv) in the same directory as the script.

Run the script using Python:

```sh
python  main.py
```

## Script Overview

The script performs the following steps:

- Importing Libraries: Imports necessary libraries such as numpy, pandas, matplotlib.pyplot, and scikit-learn modules.

- Loading Data: Loads the dataset using pandas and displays the first and last few rows of the dataset.

- Data Preprocessing:

  - Maps Gender to numerical values (0 for Male and 1 for Female).

  - Drops the User ID column.

  - Checks for missing values.

  - Splitting Data: Splits the data into training and testing sets.

  - Training Decision Tree Classifiers:

  - Trains a Decision Tree Classifier using the entropy criterion.

  - Trains a Decision Tree Classifier using the gini criterion.

  - Evaluating the Models: Evaluates the models using confusion matrix, accuracy score, and classification report.

  - Visualizing the Decision Trees: Plots and displays the decision trees.

## Visualization

The decision trees are visualized using plot_tree from scikit-learn and displayed using matplotlib.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
