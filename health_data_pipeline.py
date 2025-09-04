import pandas as pd
import numpy as np
import matplotlib
import os
from datetime import datetime

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')


class HealthDataPipeline:
    """
    Sağlık verisi için kapsamlı veri işleme ve analiz pipeline'ı
    """

    def __init__(self, data_path=None):
        """
        Pipeline'ı başlat

        Args:
            data_path (str): Veri dosyasının yolu
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_label_encoder = None
        self.pipeline_steps = []
        self.output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        # Görselleştirme ayarları
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [10, 6]


