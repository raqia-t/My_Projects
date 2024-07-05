import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load data
data = np.loadtxt("Breast_cancer_data.csv", delimiter=',', skiprows=1, dtype=float)
X = data[:, :-1]
y = data[:, -1]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train and predict with Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_test_pred_lr = lr_model.predict(x_test)

# Train and predict with Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_test_pred_rf = rf_model.predict(x_test)

# Train and predict with SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(x_train, y_train)
y_test_pred_svm = svm_model.predict(x_test)

# Define metrics calculation function
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1.0)
    recall = recall_score(y_true, y_pred, pos_label=1.0)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    specificity = TN / (TN + FP)
    return accuracy, precision, recall, specificity

# Calculate metrics for each model
test_metrics_lr = calculate_metrics(y_test, y_test_pred_lr)
test_metrics_rf = calculate_metrics(y_test, y_test_pred_rf)
test_metrics_svm = calculate_metrics(y_test, y_test_pred_svm)

# Determine best model for each metric
models_metrics = {
    "Logistic Regression": test_metrics_lr,
    "Random Forest": test_metrics_rf,
    "SVM": test_metrics_svm
}

def get_best_model(metric_index):
    best_model = None
    best_value = 0
    for model, metrics in models_metrics.items():
        if metrics[metric_index] > best_value:
            best_value = metrics[metric_index]
            best_model = model
    return best_model, best_value

best_accuracy_model, best_accuracy = get_best_model(0)
best_precision_model, best_precision = get_best_model(1)
best_recall_model, best_recall = get_best_model(2)
best_specificity_model, best_specificity = get_best_model(3)

# Create GUI using tkinter
root = tk.Tk()
root.title("Breast Cancer Prediction Model Comparison")

# Add best model labels
ttk.Label(root, text="Best Model based on Accuracy:").grid(column=0, row=0, sticky=tk.W)
ttk.Label(root, text=f"{best_accuracy_model} with Accuracy: {best_accuracy:.2f}").grid(column=1, row=0, sticky=tk.W)

ttk.Label(root, text="Best Model based on Precision:").grid(column=0, row=1, sticky=tk.W)
ttk.Label(root, text=f"{best_precision_model} with Precision: {best_precision:.2f}").grid(column=1, row=1, sticky=tk.W)

ttk.Label(root, text="Best Model based on Recall:").grid(column=0, row=2, sticky=tk.W)
ttk.Label(root, text=f"{best_recall_model} with Recall: {best_recall:.2f}").grid(column=1, row=2, sticky=tk.W)

ttk.Label(root, text="Best Model based on Specificity:").grid(column=0, row=3, sticky=tk.W)
ttk.Label(root, text=f"{best_specificity_model} with Specificity: {best_specificity:.2f}").grid(column=1, row=3, sticky=tk.W)

# Plot metrics comparison
metrics_names = ["Accuracy", "Precision", "Recall", "Specificity"]
metrics_values_lr = list(test_metrics_lr)
metrics_values_rf = list(test_metrics_rf)
metrics_values_svm = list(test_metrics_svm)

x = np.arange(len(metrics_names))
width = 0.2

fig, ax = plt.subplots()
bars_lr = ax.bar(x - width, metrics_values_lr, width, label='Logistic Regression')
bars_rf = ax.bar(x, metrics_values_rf, width, label='Random Forest')
bars_svm = ax.bar(x + width, metrics_values_svm, width, label='SVM')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars_lr)
add_labels(bars_rf)
add_labels(bars_svm)

fig.tight_layout()

# Display the plot in tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=4, columnspan=2)

root.mainloop()