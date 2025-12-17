import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Carga del Dataset de Cáncer de Mama
data = load_breast_cancer(as_frame=True)
df = data.frame
# 'target' es 0 (Maligno) y 1 (Benigno)

print("Dataset cargado. Forma (filas, columnas):", df.shape)
print("Nombres de las columnas:", df.columns.tolist())

# Renombramos la columna objetivo para mayor claridad
df = df.rename(columns={'target': 'Diagnosis'})