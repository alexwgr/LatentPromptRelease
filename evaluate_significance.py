import os
import sys
from sklearn.metrics import f1_score, accuracy_score
from statistics import mean, stdev, median
from scipy.stats import ttest_ind


if len(sys.argv) < 4:
	print('use like: evaluate_significance.py <path/to/a> <path/to/b> <num_test>')
	exit()

a_path = sys.argv[1]
b_path = sys.argv[2]
num_tests = int(sys.argv[3])

f1_positives_a = []
f1_negatives_a = []
accuracies_a = []

f1_positives_b = []
f1_negatives_b = []
accuracies_b = []


for test_num in range(num_tests):
	with open(os.path.join(a_path, 'time_vessels', str(test_num), 'test_predictions.csv')) as test_file:
		lines = test_file.readlines()
		labels_a = [[int(p) for p in line.split(',')] for line in lines]

	with open(os.path.join(b_path, 'time_vessels', str(test_num), 'test_predictions.csv')) as test_file:
		lines = test_file.readlines()
		labels_b = [[int(p) for p in line.split(',')] for line in lines]


	labels_true = [label[1] for label in labels_a]
	labels_pred_a = [label[0] for label in labels_a]
	labels_pred_b = [label[0] for label in labels_b]

	f1_positives_a.append(f1_score(y_true=labels_true, y_pred=labels_pred_a))
	f1_negatives_a.append(f1_score(y_true=labels_true, y_pred=labels_pred_a, pos_label=0))
	accuracies_a.append(accuracy_score(y_true=labels_true, y_pred=labels_pred_a))

	f1_positives_b.append(f1_score(y_true=labels_true, y_pred=labels_pred_b))
	f1_negatives_b.append(f1_score(y_true=labels_true, y_pred=labels_pred_b, pos_label=0))
	accuracies_b.append(accuracy_score(y_true=labels_true, y_pred=labels_pred_b))

print(ttest_ind(f1_positives_a, f1_positives_b, equal_var=False))
print(ttest_ind(f1_negatives_a, f1_negatives_b, equal_var=False))
print(ttest_ind(accuracies_a, accuracies_b, equal_var=False))


