from matplotlib import pyplot as plt

fprs = [0.0, 0.6, 1.0]
tprs = [0.0, 0.5, 1.0]

fig = plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fprs, tprs, 'b')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()