## My Project

I applied machine learning techniques to investigate the Titanic's passenger survival. Below is my report.

***

## Introduction 

More than 1,500 passengers lost their lives in the 1912 Titanic disaster which is now regarded as one of the most tragic maritime mishaps in history. A fascinating problem that can demonstrate the potential of machine learning is predicting a passenger's survival based on their characteristics (e.g., age, gender, class).

Based on their characteristics, passengers must be categorized as either "survived" (1) or "not survived" (0). This is important because it provides insight into patterns and factors that affected survival during the disaster. Through a supervised learning technique, we can create and evaluate machine learning models to address this issue using the Titanic dataset that is available on Kaggle.

I preprocessed the data, created meaningful features, and compared different classification models, including Support Vector Machines (SVM), Random Forests, Decision Trees, Logistic Regression, and Multi-Layer Perceptrons (MLPs). MLPs, Random Forests, and Logistic Regression made the best predictions.

## Data
The dataset contains information on passengers aboard the Titanic, with the target column being Survived (1 for survived, 0 for not survived). 

Key features include:
Numerical features: Age, Fare, SibSp (number of siblings/spouses aboard), Parch (number of parents/children aboard).
Categorical features: Pclass (ticket class), Sex, Embarked (port of embarkation).
Derived features: Cabin was transformed into a "floor" feature to represent deck levels, as the floor letter might impact survival.
Finding patterns in these characteristics that predict survival is the goal.

Here is a sample of the titanic data set
![image](https://github.com/user-attachments/assets/95e223b0-22f1-4e1e-8105-add192d29262)

Feature Engineering:

Categorical and Numerical Features:
Declared Pclass, Sex, and Embarked as categorical features.
Declared Age, SibSp, Parch, Fare, and the derived Cabin (floor) as numerical features.
Transformation of Cabin:Extracted the first letter of the Cabin value and mapped it to a numerical value representing the deck floor (e.g., A=1, B=2).
One-Hot Encoding: Transformed categorical features into binary features using one-hot encoding.

Here is a sample of the engineering data
![image](https://github.com/user-attachments/assets/4005efc3-889c-426a-aa9e-e1328133b0cd)

Correlation Matrix: Generated a heatmap to visualize correlations between features. This showed that survival was closely correlated with Pclass and Sex, among other features.
![image](https://github.com/user-attachments/assets/fcc34b39-ac0b-4f75-917b-0d6ca9030f07)

## Modelling

I compared the performance of five models: Logistic Regression, Decision Trees, Random Forests, Support Vector Machines, and Multi-Layer Perceptrons.

For the modeling section, I set the maximum iterations (max_iter) to 20,000 in the logistic regression model and MLP model. This parameter is a hyperparameter that determines the maximum number of optimization iterations the solver will perform to converge to the best solution. Setting it to a high value ensures that the model has enough iterations to find the optimal weights, especially when dealing with complex datasets. This prevents the solver from stopping prematurely before reaching convergence, which could result in suboptimal model performance.

Additionally, it is important to note that you might observe slight variations in accuracy scores if the models are run again. These differences can occur due to factors such as random initialization, the randomness involved in splitting the dataset into training and testing subsets, or stochastic processes in certain models like neural networks. Such variability underscores the importance of running multiple experiments and averaging the results to assess model performance more robustly.

1. Logistic Regression:
Logistic regression is a linear model that predicts the probability of binary outcomes. It separates classes by identifying a linear combination of properties.
Why it's appropriate: It works well and is interpretable when relationships between variables are mostly linear.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#create confusion matrix and display in graphic with numbers
from sklearn.metrics import confusion_matrix

model = LogisticRegression(max_iter=20000)
model.fit(X_Train,y_Train)
y_pred = model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("LR Accuracy:", accuracy)
```
LR Accuracy: 0.76

Confusion Matrix:
Correctly predicted both survivors and non-survivors with relatively few errors.
![image](https://github.com/user-attachments/assets/54722b77-b220-4402-abf0-4bf6d74374a5)
Strength: Simple, interpretable, and performs well on structured data.

2. Decision Trees:
Decision trees split the data into branches based on feature thresholds. Although they are simple to comprehend, they are prone to overfitting.
Why it's appropriate: useful for identifying patterns that are not linear, also makes very clear the method that the model is using for its decisions.

```python
from sklearn.tree import DecisionTreeClassifier

dc_model = DecisionTreeClassifier(random_state=314159)
dc_model.fit(X_Train, y_Train)
y_pred = dc_model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("DT Accuracy:", accuracy)
```
DT Accuracy: 0.77

Confusion Matrix:
Slightly higher overfitting compared to Logistic Regression.
![image](https://github.com/user-attachments/assets/a4ebdfd6-af18-4458-9476-7c50ec703092)
Strength: Captures non-linear patterns but prone to overfitting without pruning.

3. Random Forest:
A random forest is an ensemble of decision trees, which reduces overfitting and averages forecasts to increase accuracy.
Why it's appropriate: combines low variance with excellent accuracy.

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=314159,max_depth=3)
rf_model.fit(X_Train,y_Train)
y_pred = model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("RF Accuracy:", accuracy)

#lets display the best tree
from sklearn.tree import export_graphviz
import graphviz

best_tree = rf_model.estimators_[0]
dot_data = export_graphviz(best_tree, out_file=None, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("random_forest_tree")
```
RF Accuracy: 0.76

Confusion Matrix:
Balanced predictions with fewer errors across both classes.
![image](https://github.com/user-attachments/assets/f49683ef-9ddf-412b-80f2-710240d2ed37)
Strength: Reduces overfitting and provides feature importance.

The Random Forest Best Fit Tree Visualization makes it easy to understand how the model makes decisions. The random forest model determined whether the passenger was in third class (Pclass = 3) is the most important factor affecting survivability, and this is where the tree starts. Since it had the biggest influence on survival prediction, this feature acts as the primary decision point.

The left branch of the model indicates a better chance of survival if the passenger was not in third class (Pclass ≠ 3).
After that, the following choice usually considers sex, where being female greatly improves survival odds.
With each split improving the prediction, the tree keeps branching according to other characteristics like Age or Fare.

In contrast, the model follows the right branch, where survival chances decrease, if the passenger is in third class (Pclass = 3). Being female or younger, however, has a positive impact on survival even in this group.

The final predictions, which indicate whether the passenger survived (1) or not (0), are displayed at the leaf nodes of the tree. According to the visual, non-third-class passengers—particularly women—had a significantly greater survival percentage, which is consistent with historical accounts of the Titanic tragedy. This hierarchical method demonstrates how well decision trees capture the significance of features and the logical evolution of survival predictions.

Best Fit Tree Visualization:
![image](https://github.com/user-attachments/assets/4ed8f5f7-cd9e-49d3-8e7e-39227e72a8f6)



4. Support Vector Machines:
SVMs use an ideal hyperplane to categorize data. To handle non-linear decision boundaries, we used a polynomial kernel. Why it's appropriate: useful for intricate connections in smaller datasets.

```python
from sklearn.svm import SVC

svc_model = SVC(kernel='poly', degree=4, probability=True)
svc_model.fit(X_Train, y_Train)
y_pred = model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("SVM Accuracy:", accuracy)
```
SVM Accuracy: 0.76

Confusion Matrix:
![image](https://github.com/user-attachments/assets/60446ddf-2f13-4e39-9d53-f832aa11bffc)
Strength: Captures complex patterns but computationally expensive for large datasets.



5. Neural Networks (MLP):
An MLP is a type of neural network that can recognize hierarchical patterns.
Why it's appropriate: able to simulate intricate non-linear interactions found in the dataset.

```python
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=2000)
mlp_model.fit(X_Train, y_Train)
y_pred = mlp_model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("MLP Accuracy:", accuracy)
```
MLP Accuracy: 0.76

Confusion Matrix:
Slightly higher accuracy, with robust handling of non-linearity.
![image](https://github.com/user-attachments/assets/57a8f110-00b5-44a7-9bde-eb0895665876)
Strength: Powerful, but requires more computational resources and fine-tuning.

For the Multilayer Perceptron (MLP) model, I chose a specific arrangement of hidden layers with 16 nodes in the first hidden layer and 8 nodes in the second hidden layer. The reasoning behind this configuration is as follows:

The first hidden layer contains 16 nodes because 16 is the nearest power of 2 to the number of features (14) without going under it. Using powers of 2 is a common practice in neural network design, as these configurations often yield better performance due to optimization efficiency in underlying computational processes.

The second hidden layer contains 8 nodes, which is half the size of the first layer. This reduction creates a "bridging" layer that computes the relationships between features learned in the first layer and prepares the data for the 2-node output layer, which represents the binary classification task (survival or not). The reduced size also helps prevent overfitting while retaining the model's ability to learn complex relationships between features.


## Results

Comparison of Models:
Logistic Regression, Random Forest, and MLP performed similarly, achieving the highest accuracy scores (~0.76-0.77).
Decision Trees and SVMs performed slightly worse, likely due to overfitting or sensitivity to parameter tuning.

ROC Curve:
The ROC curve demonstrated that Logistic Regression, Random Forest, and MLP had the highest AUC values.

![image](https://github.com/user-attachments/assets/1f5baa48-a14e-4027-88e7-fd824c760e7a)
Start from the top left corner, the line closest to the corner is the best model with recall and precision.


## Discussion

With similar accuracy, the top three models were MLP, Random Forest, and Logistic Regression. Random Forest is the most interpretable, making it suitable for explaining the results to non-technical audiences. Random Forest offers strong performance and insights into feature importance, but MLP successfully captured non-linear correlations.

Looking at the confusion matrices for both the Random Forest and Logistic Regression models, we can see that they are identical. This might suggest that reviewing the tree from the Random Forest could not only explain how the Random Forest made its decisions but also provide insights into how the Logistic Regression evaluated each passenger. Random Forest and Logistic Regression had similar scores because both made effective use of the most crucial characteristics (Sex, Pclass, Age, etc.). Overfitting caused Decision Trees to perform slightly worse, and SVM had trouble with parameter sensitivity.

## Conclusion

Several machine learning models were used in this project to predict Titanic survival. The most successful ones were random forests, logistic regression, and neural networks. The Random Forest classifier was the most effective model among them, striking a balance between interpretability and accuracy. Age, Pclass, and sex were the most significant factors affecting survival, demonstrating the effectiveness of machine learning in resolving categorization issues. Additionally, this project provides a foundation for additional research utilizing ensemble models or hyperparameter tuning.

From this work, the following conclusions can be made:
1. Passenger class, gender, and age are the most important predictors of survival.
2. Ensemble methods, like random forests, excel in balancing accuracy and variance, making them a robust choice for classification problems.

Here is how this work could be developed further in a future project:
Examine how feature engineering and selection affect performance, and use domain expertise to improve predictors.
To further increase prediction accuracy, investigate deep learning models with increasingly sophisticated architectures.
To generalize findings and broaden the use of these techniques, apply comparable models to additional datasets with survival-based outcomes, such as healthcare or disaster aid datasets.

## References
Titanic, Machine Learning from Disaster (https://www.kaggle.com/c/titanic/data)
