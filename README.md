> Making this repository to complete the mid-semester exam for the Machine Learning Course with lecturer Mr. Mohammad Hamim Zajuli Al Faroby, S.Sc., M.Mat. The hope is that this repository will be helpful in studying Unsupervised Learning.

# Unsupervised-Learning-Method
SOM (Self-organizing maps), PCA (Principal component analysis), K-means Clustering

| Metode                      | Pengertian                                                                                                     |
|-----------------------------|---------------------------------------------------------------------------------------------------------------|
| Self-Organizing Map (SOM)   | Metode pembelajaran tak terawasi yang mengorganisir data dalam bentuk grid atau peta, mempertahankan struktur topologis.|
| K-means Clustering          | Algoritma pengelompokan yang membagi data menjadi k kelompok (cluster) berdasarkan pusat cluster yang diinisiasi secara acak.|
| Principal Component Analysis (PCA) | Metode reduksi dimensi yang mengidentifikasi komponen utama yang membawa sebagian besar varians dalam data. |

# Metadata
| Nomor | Kolom               | Deskripsi                                                                                                    |
|:-------:|:---------------------|:--------------------------------------------------------------------------------------------------------------|
| 1     | Id                  | Id dari masing-masing pasien                                                                                 |
| 2     | Age                 | Usia dari pasien                                                                                             |
| 3     | Gender              | Jenis kelamin dari pasien                                                                                    |
| 4     | Paracetamol         | Obat untuk menurunkan demam serta meredakan nyeri ringan hingga sedang.                                       |
| 5     | Acetaminophen       | Kata lain dari paracetamol, obat untuk menurunkan demam serta meredakan nyeri ringan hingga sedang.          |
| 6     | Aspirin             | Obat pereda nyeri, penurun demam, dan antiinflamasi.                                                          |
| 7     | Ibuprofen           | Digunakan untuk meredakan nyeri, peradangan, dan demam.                                                        |
| 8     | Tylenol             | Nama lain untuk paracetamol (acetaminophen), digunakan untuk meredakan nyeri dan menurunkan demam.            |
| 9     | Diphen              | Digunakan untuk meredakan gejala alergi seperti pilek, bersin, dan gatal-gatal.                               |
| 10    | Bronkaid            | Digunakan dengan guaifenesin untuk meredakan gejala asma dan penyakit paru obstruktif kronis (PPOK).          |
| 11    | Aprocline           | Digunakan untuk meredakan hidung tersumbat dan gejala pilek.                                                  |
| 12    | Pseudophetrine      | Sama seperti aprocline, digunakan untuk meredakan hidung tersumbat dan gejala pilek.                           |
| 13    | Aleve               | Nama lain untuk naproxen, digunakan untuk meredakan nyeri dan peradangan, sering digunakan untuk mengobati arthritis.|
| 14    | Naproxen            | Digunakan untuk meredakan nyeri dan peradangan, sering digunakan untuk mengobati arthritis.                   |
| 15    | Proprinal           | Serupa dengan ibuprofen, digunakan untuk meredakan nyeri dan peradangan.                                      |
| 16    | Addaprin            | Merek dagang untuk ibuprofen, tergolong dalam kategori NSAID (Antiinflamasi nonsteroid).                      |
| 17    | Zebutal             | Digunakan untuk meredakan sakit kepala tension dengan kombinasi butalbital, asam asetosal, dan kafein.       |
| 18    | Meperdine           | Digunakan untuk meredakan nyeri sedang hingga berat, termasuk pascaoperasi.                                    |
| 19    | Abenol              | Merek dagang untuk paracetamol atau acetaminophen.                                                            |
| 20    | Actiprofen          | Digunakan untuk meredakan nyeri dan peradangan, termasuk arthritis.                                            |
| 21    | Vazalore            | Sama seperti aspirin, digunakan untuk meredakan nyeri, peradangan, dan demam.                                  |
| 22    | Year                | Tahun dari data pasien di-inputkan                                                                           |

# Import Library
```python
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from sklearn_som.som import SOM
from minisom import MiniSom

import time

%matplotlib inline
sns.set(color_codes=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
```

# Simple Way to use
## Self-Organizing Map (SOM)
```python
from sklearn.cluster import KMeans #Import
som_shape = (1, 5)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=0.5, learning_rate=0.5)
som.train(data_std, 1000, verbose=True)
```

# K-means Clustering
```python
from sklearn.cluster import KMeans #Import
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_std)
```
# Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA #Import
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_std)
```
