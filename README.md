# 🧠 Time Series Classification – ANNDL Challenge 2

This repository contains our work for [**Artificial Neural Networks and Deep Learning – Challenge 2 (2022)**](https://codalab.lisn.upsaclay.fr/competitions/9056#learn_the_details).  
The task was to build a deep learning model capable of classifying multivariate **time series into 12 classes**.  
Our final approach achieved an accuracy of **73.38% on the Codalab test set**, and ranked **90th out of 525** participants **(top 17%)** on the leaderboard.

---

## 📂 Repository contents

We included in the delivery folder all the most relevant approaches, features, and architectures implemented during the challenge:

- **`DELIVERY_Aielli_Buono_Buranti.ipynb`**  
  Notebook of our **best model** according to Codalab test evaluation. Requires the included **`scaler`** for correct execution.

- **`tsaug_ensemble_method.ipynb`**  
  Experiments with **data augmentation** using the `tsaug` library.

- **`weighted_sum_ensemble.ipynb`**  
  Attempts to optimize the **weight vector** for an ensemble of multiple models.

---

## 📝 Project Overview

### 🔹 Dataset
- Multivariate time series with **36 timesteps** and **6 features** per sample.  
- **12 target classes**.  
- Split: **80/10/10** for train/validation/test.  
- Stratified splits to preserve class distribution.

### 🔹 Preprocessing
- **Standardization** using `RobustScaler` (median + quantile scaling).  
- **Oversampling** to mitigate class imbalance.  
- **Data augmentation**:  
  - Custom noise, scaling, shifting, cropping (did not improve).  
  - `tsaug` library for cropping subsequences, random quantization, drifting, and reversing.  
- **Feature reduction** (based on correlation with target) → did not improve results.  
- **Window sliding** → improved local accuracy but led to overfitting on Codalab.

### 🔹 Architectures explored
- **Baseline**: LSTM, BiLSTM, Conv1D.  
- Tuned with deeper layers, dropout, hidden units.  
- **Best single model**: Conv1D + BiLSTM hybrid.  
- **Attention layers** were tested but dropped due to high complexity and limited benefit.

### 🔹 Ensemble methods
- Combined multiple optimized models (LSTM, BiLSTM, Conv1D, GRU) to leverage their complementary strengths.  
- Achieved **+4% accuracy** over single models, and ~8–10% improvements in precision/recall.  
- Weighted-sum ensembles were explored, but limited by small validation set size.

---

## 📊 Results
- **Leaderboard:** ranked **90th out of 525** participants **(top 17%)**.
- **Single best model**: Conv1D + BiLSTM  
- **Final ensemble**: Hybrid (LSTM + BiLSTM + Conv1D + GRU)  
- **Codalab Test Accuracy**: **73.38%**  

---

## 📚 Technologies
- Python, NumPy, Pandas, Scikit-learn  
- TensorFlow / Keras  
- tsaug (for time series augmentation)  
- Google Colab (for training & experiments)

---

## 👥 Authors
- Riccardo Luigi Aielli  
- Salvatore Buono  
- Simone Buranti
