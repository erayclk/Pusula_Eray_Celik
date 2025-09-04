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

        def load_data(self, data_path=None):
            """
            Veri dosyasını yükle

            Args:
                data_path (str): Veri dosyasının yolu
            """
            if data_path:
                self.data_path = data_path

            try:
                if self.data_path.endswith('.xlsx'):
                    self.raw_data = pd.read_excel(self.data_path)
                elif self.data_path.endswith('.csv'):
                    self.raw_data = pd.read_csv(self.data_path)
                else:
                    raise ValueError("Desteklenen dosya formatları: .xlsx, .csv")

                print(f"Veri başarıyla yüklendi. Boyut: {self.raw_data.shape}")
                self.pipeline_steps.append("Veri yüklendi")
                return self.raw_data

            except Exception as e:
                print(f"Veri yükleme hatası: {e}")
                return None




