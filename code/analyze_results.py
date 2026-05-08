#!/usr/bin/env python3
import pandas as pd
import sys

df = pd.read_csv('/home/ansari/Desktop/Dr Ghosh/Feature Selection/V2/results/pipeline_results.csv')
methods = ['MI','Fisher','DDFF-L1','DDFF-L2','DDFF-KL','DDFF-Max','DDFF-Ensemble']
lines = []
lines.append('PEAK kNN ACCURACY')
lines.append(f'{"Dataset":15s} | ' + ' | '.join(f'{m:13s}' for m in methods))
lines.append('-'*130)
for ds in df.dataset.unique():
    vals = []
    for m in methods:
        sub = df[(df.dataset==ds)&(df.method==m)]
        knn = sub.groupby('k')['knn_accuracy'].mean()
        pk = knn.max()
        bk = knn.idxmax()
        sd = sub[sub.k==bk]['knn_accuracy'].std()
        vals.append(f'{pk:5.1f}+/-{sd:4.1f} ')
    lines.append(f'{ds:15s} | ' + ' | '.join(vals))

lines.append('')
lines.append('PEAK SVM ACCURACY')
lines.append(f'{"Dataset":15s} | ' + ' | '.join(f'{m:13s}' for m in methods))
lines.append('-'*130)
for ds in df.dataset.unique():
    vals = []
    for m in methods:
        sub = df[(df.dataset==ds)&(df.method==m)]
        svm = sub.groupby('k')['svm_accuracy'].mean()
        pk = svm.max()
        bk = svm.idxmax()
        sd = sub[sub.k==bk]['svm_accuracy'].std()
        vals.append(f'{pk:5.1f}+/-{sd:4.1f} ')
    lines.append(f'{ds:15s} | ' + ' | '.join(vals))

lines.append('')
lines.append('BEST k FOR kNN')
lines.append(f'{"Dataset":15s} | ' + ' | '.join(f'{m:13s}' for m in methods))
lines.append('-'*130)
for ds in df.dataset.unique():
    vals = []
    for m in methods:
        sub = df[(df.dataset==ds)&(df.method==m)]
        knn = sub.groupby('k')['knn_accuracy'].mean()
        bk = knn.idxmax()
        vals.append(f'    k={bk:<8}')
    lines.append(f'{ds:15s} | ' + ' | '.join(vals))

with open('/home/ansari/Desktop/Dr Ghosh/Feature Selection/V2/results/analysis_summary.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
sys.exit(0)
