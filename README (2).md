# 📡 Telco Customer Churn Prediction — ANN

Telekommunikasiya şirkətinin müştəri məlumatları əsasında **müştəri çıxışını (churn)** proqnozlaşdıran end-to-end **Süni Neyron Şəbəkəsi (ANN)** layihəsi.

---

## 📊 Dataset

**Mənbə:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| | |
|---|---|
| Müştəri sayı | 7,043 |
| Xüsusiyyət sayı | 21 (preprocessing sonrası 28) |
| Hədəf dəyişən | `Churn` — 0: Qalır, 1: Çıxır |
| Sinif balansı | ~74% qalır / ~26% çıxır (imbalanced) |

---

## 🔄 Pipeline

### 1. 📥 Məlumatların Yüklənməsi
Kaggle API ilə dataset yüklənir, CSV oxunur.

### 2. 🔍 Kəşfiyyat Analizi (EDA)
- Ümumi baxış — `shape`, `info`, `describe`, `isnull`
- **Korrelyasiya xəritəsi** — sütunlar arası əlaqə
- **Boxplot** — `tenure` və `MonthlyCharges` vs Churn
- **Countplot** — bütün kateqorial sütunların churn üzrə paylanması
- **Histplot + KDE** — ədədi sütunların paylanması

### 3. 🛠️ Preprocessing

| Addım | Ətraflı |
|---|---|
| `customerID` silindi | Model üçün mənasızdır |
| `TotalCharges` silindi | `tenure` ilə 0.83 korrelyasiya — multicollinearity |
| `gender` silindi | Churn üzrə eyni nisbət — fərqsiz feature |
| Outlier capping | `tenure`, `MonthlyCharges` → [1%, 99%] |
| Binary encoding | `Yes/No` sütunları → `1/0` |
| One-Hot Encoding | Çoxkateqoriyalı sütunlar — `get_dummies` |
| StandardScaler | `tenure`, `MonthlyCharges` normallaşdırıldı |
| Train/Test split | 80% / 20%, `random_state=42` |

### 4. 🧠 Model Arxitekturası

```
Input(28)
  → Dense(32, ReLU) → Dropout(0.1)
  → Dense(16, ReLU) → Dropout(0.15)
  → Dense(1, Sigmoid)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Metrics:** Accuracy, Precision, Recall
- **EarlyStopping:** `patience=3`, `val_loss` izlənilir, ən yaxşı çəkilər bərpa edilir
- **Class Weight:** `compute_class_weight('balanced')` — imbalanced dataset problemi həll edildi

### 5. 📈 Qiymətləndirmə
- `model.evaluate()` — test seti üzərində
- `classification_report` — sinif üzrə dəqiq göstəricilər
- **Confusion Matrix** — heatmap ilə vizuallaşdırma
- **Loss / Accuracy / Precision / Recall qrafikleri** — epoch boyunca train vs validation

---

## 📉 Model Nəticələri

| Metrika | Dəyər |
|---|---|
| Test Accuracy | ~75–76% |
| Test Precision | ~52–53% |
| Test Recall | ~81–83% |

> Recall-un yüksək olması əsas məqsəddir — şirkət üçün vacib olan **çıxacaq müştərini vaxtında tanımaqdır.**  
> Precision-un aşağı olması `class_weight` tətbiqinin gözlənilən nəticəsidir.

---

## ⚙️ Tələblər

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

---

## 🚀 İşə Salma

```bash
# 1. Kaggle API key ~/.kaggle/kaggle.json-da olmalıdır
# 2. Notebooku aç və addım-addım işlət
jupyter notebook telco_customer_churn_with_ANN.ipynb
```

---

## 📂 Fayl Strukturu

```
.
├── telco_customer_churn_with_ANN.ipynb   # Əsas notebook
└── README.md
```
