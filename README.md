# 🌫️ Tabriz Air Pollution Prediction

This project was developed as part of the NTI AI Training Program. It uses Machine Learning to predict **PM2.5 air pollution levels** based on environmental measurements collected from multiple air quality monitoring stations in Tabriz, Iran. The project includes data preprocessing, model training, evaluation, and a Streamlit web application for interactive predictions.

---

# 📊 Dataset Description

The project uses the **Tabriz Pollution Dataset**, which combines data from three monitoring stations:

- Abrasan
- Bashumal
- RastaKucha

### Data Preprocessing
- Merged the three datasets into one dataset.
- Removed unnecessary columns such as **Time**.
- Handled missing values and prepared the data for training.
- Selected **PM2.5** as the target variable.
- Standardized the input features using **StandardScaler**.

---

# 📁 Project Structure

```
Tabriz-Air-Pollution/
│
├── Dataset/
│   ├── Abrasan.csv
│   ├── Bashumal.csv
│   └── RastaKucha.csv
│
├── Models/
│   ├── airPollution_model.pkl
│   └── scaler.pkl
│
├── streamlit_app.py
├── nti-ai-project-yousef-mohamed-yousef-ismail.ipynb
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tabriz-air-pollution.git
```

### 2. Navigate to the project directory

```bash
cd tabriz-air-pollution
```

### 3. (Optional) Create a virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run the Project

### Run the Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and run all cells.

### Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

# 🤖 Model Description

The project trains a **Random Forest Classifier** to predict PM2.5 pollution levels.

### Model Pipeline

- Load and merge the three datasets.
- Remove unnecessary columns.
- Split features and target.
- Split the dataset into training and testing sets (80% / 20%).
- Standardize the features using **StandardScaler**.
- Train a **Random Forest Classifier** with:
  - 300 decision trees
  - Maximum depth = 15
  - Minimum samples split = 5
- Evaluate the model using classification metrics.
- Save the trained model as:

```
airPollution_model.pkl
```

and the scaler as:

```
scaler.pkl
```

---

# 🚀 Streamlit Deployment Link

**Live Demo**

> https://your-streamlit-app.streamlit.app

*(Replace this placeholder with your deployed Streamlit application URL.)*

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

---

# 👤 Author

**Yousef Mohamed**

NTI AI Training Program
