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

    def explore_data(self):
        """
        Veri keşfi yap
        """
        if self.raw_data is None:
            print("Önce veri yükleyin!")
            return

        print("=== VERİ KEŞFİ ===")
        print(f"Veri boyutu: {self.raw_data.shape}")
        print(f"Toplam bellek kullanımı: {self.raw_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        print("\n=== VERİ TİPLERİ ===")
        print(self.raw_data.dtypes)

        print("\n=== EKSİK VERİ ANALİZİ ===")
        missing_data = self.raw_data.isnull().sum()
        missing_percent = (missing_data / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Eksik Veri Sayısı': missing_data,
            'Oran (%)': missing_percent
        })
        print(missing_df[missing_df['Eksik Veri Sayısı'] > 0])

        print("\n=== SAYISAL SÜTUNLAR İÇİN İSTATİSTİKLER ===")
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(self.raw_data[numeric_cols].describe())

        self.pipeline_steps.append("Veri keşfi tamamlandı")

    def visualize_data(self, max_pairplot_features=5):
        """
        EDA görselleştirmeleri üret ve kaydet (histogramlar, korelasyon ısı haritası, sınırlı pairplot)
        """
        if self.raw_data is None:
            print("Önce veri yükleyin!")
            return

        print("=== EDA GÖRSELLEŞTİRMELERİ OLUŞTURULUYOR ===")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Histogramlar (sayısal sütunlar)
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            try:
                ax = self.raw_data[numeric_cols].hist(bins=20, figsize=(14, 10))
                plt.suptitle('Sayısal Sütunlar - Histogramlar')
                hist_path = os.path.join(self.output_dir, f'histograms_{timestamp}.png')
                plt.tight_layout()
                plt.savefig(hist_path)
                plt.close()
                print(f"Histogram görseli kaydedildi: {os.path.abspath(hist_path)}")
            except Exception as e:
                print(f"Histogram oluşturulamadı: {e}")

        # Korelasyon ısı haritası
        if numeric_cols and len(numeric_cols) >= 2:
            try:
                corr = self.raw_data[numeric_cols].corr(numeric_only=True)
                plt.figure(figsize=(12, 8))
                sns.heatmap(corr, cmap='coolwarm', center=0)
                plt.title('Korelasyon Isı Haritası')
                corr_path = os.path.join(self.output_dir, f'correlation_heatmap_{timestamp}.png')
                plt.tight_layout()
                plt.savefig(corr_path)
                plt.close()
                print(f"Korelasyon ısı haritası kaydedildi: {os.path.abspath(corr_path)}")
            except Exception as e:
                print(f"Korelasyon ısı haritası oluşturulamadı: {e}")

        # Pairplot (sınırlı sayıda özellik)
        if numeric_cols:
            try:
                selected = numeric_cols[:max_pairplot_features]
                sns.pairplot(self.raw_data[selected], corner=True)
                pairplot_path = os.path.join(self.output_dir, f'pairplot_{timestamp}.png')
                plt.savefig(pairplot_path)
                plt.close()
                print(f"Pairplot kaydedildi: {os.path.abspath(pairplot_path)}")
            except Exception as e:
                print(f"Pairplot oluşturulamadı: {e}")

        self.pipeline_steps.append("EDA görselleştirmeleri oluşturuldu")

    def clean_data(self):
        """
        Veri temizliği yap
        """
        print("=== VERİ TEMİZLİĞİ BAŞLADI ===")

        # Kopya oluştur
        self.processed_data = self.raw_data.copy()

        # Eksik veri doldurma stratejileri
        self._fill_missing_values()

        # Veri tipi dönüşümleri
        self._convert_data_types()

        # Duplicate kayıtları temizle
        initial_rows = len(self.processed_data)
        self.processed_data = self.processed_data.drop_duplicates()
        final_rows = len(self.processed_data)

        if initial_rows != final_rows:
            print(f"Duplicate kayıtlar temizlendi: {initial_rows - final_rows} kayıt kaldırıldı")

        print("Veri temizliği tamamlandı!")
        self.pipeline_steps.append("Veri temizliği tamamlandı")

    def _fill_missing_values(self):
        """
        Eksik verileri doldur
        """
        # Kategorik sütunlar için mode kullan
        categorical_cols = self.processed_data.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if self.processed_data[col].isnull().any():
                mode_value = self.processed_data[col].mode()[0]
                self.processed_data[col].fillna(mode_value, inplace=True)
                print(f"'{col}' sütunundaki eksik veriler '{mode_value}' ile dolduruldu")

        # Sayısal sütunlar için median kullan
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.processed_data[col].isnull().any():
                median_value = self.processed_data[col].median()
                self.processed_data[col].fillna(median_value, inplace=True)
                print(f"'{col}' sütunundaki eksik veriler median ({median_value}) ile dolduruldu")


