import os
import sys
from sklearn.metrics import f1_score, accuracy_score
from statistics import mean, stdev, median


if len(sys.argv) < 4:
	print('use like: evaluate_metrics.py <path/to/time_vessels> <num_test> <predictions_file>')
	exit()

time_vessels_path = sys.argv[1]
num_tests = int(sys.argv[2])
predictions_file = sys.argv[3]


f1_positives = []
f1_negatives = []
accuracies = []

for test_num in range(num_tests):
	with open(os.path.join(time_vessels_path, 'time_vessels', str(test_num), predictions_file)) as test_file:
		lines = test_file.readlines()
		labels = [[int(p) for p in line.split(',')] for line in lines]


	labels_true = [label[0] for label in labels]
	labels_pred = [label[1] for label in labels]

	f1_positives.append(f1_score(y_true=labels_true, y_pred=labels_pred))
	f1_negatives.append(f1_score(y_true=labels_true, y_pred=labels_pred, pos_label=0))
	accuracies.append(accuracy_score(y_true=labels_true, y_pred=labels_pred))

print('F1 pos: {} ({}) : {}'.format(mean(f1_positives), stdev(f1_positives), median(f1_positives)))
print('F1 neg: {} ({}) : {}'.format(mean(f1_negatives), stdev(f1_negatives), median(f1_negatives)))
print('Accuracy: {} ({}) : {}'.format(mean(accuracies), stdev(accuracies), median(accuracies)))




