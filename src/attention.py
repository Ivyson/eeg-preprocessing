import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

src_data = "datasets/acquiredDataset.csv"
model_path = "models/attention.joblib"
dataframe = pd.read_csv(src_data)
results = []

for val in dataframe["attention"]:
    if val >= 50:
        results.append("1")
    else:
        results.append("0")
dataframe["result"] = results
print(
    f"You Currently have {dataframe['classification'].value_counts()} Classifications"
)
y = dataframe.pop("classification")
scaler = StandardScaler()
X = scaler.fit_transform(dataframe)
dataframe = dataframe.drop(columns=["attention"])  # Removes the attention column
print(f"Columns in dataframe: {dataframe.columns.tolist()}")
print(f"Shape: {dataframe.shape}")
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=46
)  # Have a 70-30 train
# to test data
# Random state 46,
# gives 73% accuracy
# Load the model to the permanent storage


knn = KNeighborsClassifier()
k_range = [
    x for x in range(1, 35) if x % 2 > 0
]  # On 35. we are getting max accuracy of: 71%
# print(k_range)
parameter_grid = dict(n_neighbors=k_range)

grid = GridSearchCV(
    knn,
    param_grid=parameter_grid,
    cv=10,
    scoring="accuracy",
    return_train_score=False,
    verbose=1,
)


# Now, Train the model on the splitted data
grid_model = grid.fit(X=x_train, y=y_train)
accuracy = grid_model.best_score_ * 100
print(f"The Accuracy of the trained model : {accuracy}")
print("Testing the model Performance")
y_predicted = grid_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred=y_predicted)
print(cm)
if accuracy > 71:
    dump(grid_model, "models/attention_dataset.joblib")
    print("Saving te model")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_model.classes_)
disp.plot(cmap=plt.cm.Blues)  # Optional: add a colormap for nicer visuals
plt.title("Confusion Matrix for KNN Classifier")
plt.savefig("figures/attention_dataset_cm.png")
plt.show()
