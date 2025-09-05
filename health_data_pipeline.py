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

    def _convert_data_types(self):
        """
        Veri tiplerini dönüştür
        """
        # TedaviSuresi ve UygulamaSuresi sütunlarını temizle ve sayıya çevir
        if 'TedaviSuresi' in self.processed_data.columns:
            self.processed_data['TedaviSuresi'] = self.processed_data['TedaviSuresi'].astype(str).str.extract(
                r'(\d+)').astype(float)

        if 'UygulamaSuresi' in self.processed_data.columns:
            self.processed_data['UygulamaSuresi'] = self.processed_data['UygulamaSuresi'].astype(str).str.extract(
                r'(\d+)').astype(float)

    def feature_engineering(self, preserve_columns=None):
        """
        Özellik mühendisliği yap

        Args:
            preserve_columns (list): One-hot encoding'den korunacak sütunlar
        """
        if self.processed_data is None:
            print("Önce veri temizliği yapın!")
            return

        print("=== ÖZELLİK MÜHENDİSLİĞİ BAŞLADI ===")

        # Korunacak sütunları belirt
        if preserve_columns is None:
            preserve_columns = []

        # Alerji sütununu one-hot encoding ile dönüştür
        if 'Alerji' in self.processed_data.columns:
            alerji_dummies = self.processed_data['Alerji'].str.get_dummies(sep=',')
            alerji_dummies = alerji_dummies.add_prefix('Alerji_')
            self.processed_data = pd.concat([self.processed_data, alerji_dummies], axis=1)
            self.processed_data.drop('Alerji', axis=1, inplace=True)
            print("Alerji sütunu one-hot encoding ile dönüştürüldü")

        # Kategorik sütunları one-hot encoding ile dönüştür (korunacak sütunlar hariç)
        categorical_cols = self.processed_data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in preserve_columns]

        if len(categorical_cols) > 0:
            self.processed_data = pd.get_dummies(self.processed_data, columns=categorical_cols, drop_first=True)
            print(f"Kategorik sütunlar one-hot encoding ile dönüştürüldü: {list(categorical_cols)}")

        # Gereksiz sütunları kaldır
        cols_to_drop = ['HastaNo', 'Tanilar', 'TedaviAdi', 'UygulamaYerleri', 'KronikHastalik']
        existing_cols_to_drop = [col for col in cols_to_drop if col in self.processed_data.columns]

        if existing_cols_to_drop:
            self.processed_data.drop(columns=existing_cols_to_drop, inplace=True)
            print(f"Gereksiz sütunlar kaldırıldı: {existing_cols_to_drop}")

        print("Özellik mühendisliği tamamlandı!")

        self.pipeline_steps.append("Özellik mühendisliği tamamlandı")

    def scale_features(self):
        """
        Sayısal özellikleri ölçeklendir
        """
        if self.processed_data is None:
            print("Önce özellik mühendisliği yapın!")
            return

        print("=== ÖZELLİK ÖLÇEKLENDİRME BAŞLADI ===")

        # Sayısal sütunları bul
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            # StandardScaler uygula
            self.processed_data[numeric_cols] = self.scaler.fit_transform(self.processed_data[numeric_cols])
            print(f"Sayısal sütunlar standartlaştırıldı: {list(numeric_cols)}")

        self.pipeline_steps.append("Özellik ölçeklendirme tamamlandı")

    def prepare_for_modeling(self, target_column=None):
        """
        Modelleme için veriyi hazırla

        Args:
            target_column (str): Hedef değişken sütunu
        """
        if self.processed_data is None:
            print("Önce veri işleme adımlarını tamamlayın!")
            return None, None

        print("=== MODELLEME HAZIRLIĞI ===")

        # Hedef değişken belirtilmemişse, ilk sayısal sütunu kullan
        if target_column is None:
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_column = numeric_cols[0]
                print(f"Hedef değişken otomatik seçildi: {target_column}")

        if target_column not in self.processed_data.columns:
            print(f"Hedef sütun bulunamadı: {target_column}")
            return None, None

        # Hedef değişkeni ayır
        X = self.processed_data.drop(columns=[target_column])
        y = self.processed_data[target_column]

        print(f"Özellik matrisi boyutu: {X.shape}")
        print(f"Hedef değişken boyutu: {y.shape}")

        self.pipeline_steps.append("Modelleme hazırlığı tamamlandı")
        return X, y

    def train_model(self, X, y, test_size=0.2, random_state=42, model_type='auto'):
        """
        Model eğitimi yap

        Args:
            X: Özellik matrisi
            y: Hedef değişken
            test_size: Test seti oranı
            random_state: Rastgelelik için seed
            model_type: Model tipi ('auto', 'classification', 'regression')
        """
        print("=== MODEL EĞİTİMİ BAŞLADI ===")

        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Eğitim seti boyutu: {X_train.shape}")
        print(f"Test seti boyutu: {X_test.shape}")

        # Hedef değişken tipini otomatik belirle
        if model_type == 'auto':
            if y.dtype in ['object', 'bool'] or len(y.unique()) < 20:
                model_type = 'classification'
                print("Hedef değişken tipi otomatik belirlendi: Sınıflandırma")
            else:
                model_type = 'regression'
                print("Hedef değişken tipi otomatik belirlendi: Regresyon")

        # Model tipine göre uygun modeli seç
        if model_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, confusion_matrix

            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)

            # Tahmin yap
            y_pred = model.predict(X_test)

            # Model performansını değerlendir
            print("\n=== SINIFLANDIRMA MODEL PERFORMANSI ===")
            print(classification_report(y_test, y_pred))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Gerçek Değerler')
            plt.xlabel('Tahmin Edilen Değerler')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'confusion_matrix_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Confusion matrix görseli kaydedildi: {os.path.abspath(plot_path)}")

        else:  # regression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)

            # Tahmin yap
            y_pred = model.predict(X_test)

            # Model performansını değerlendir
            print("\n=== REGRESYON MODEL PERFORMANSI ===")
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R² Score: {r2:.4f}")

            # Gerçek vs Tahmin grafiği
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Gerçek Değerler')
            plt.ylabel('Tahmin Edilen Değerler')
            plt.title('Gerçek vs Tahmin Değerleri')
            plt.grid(True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'regression_actual_vs_pred_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Regresyon görseli kaydedildi: {os.path.abspath(plot_path)}")

        self.pipeline_steps.append("Model eğitimi tamamlandı")
        return model, X_train, X_test, y_train, y_test

    def _build_sklearn_preprocessing(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Sklearn tabanlı ön işleme pipeline'ı oluştur: KNNImputer + StandardScaler (sayısal),
        most_frequent imputation + OneHotEncoder (kategorik)
        """
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_transformer = SkPipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = SkPipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='drop'
        )

        return preprocessor

    def train_with_sklearn_pipeline(self, X, y, test_size=0.2, random_state=42, model_type='auto'):
        """
        Sklearn ColumnTransformer + Pipeline kullanarak uçtan uca eğitim yap.
        Sınıflandırmada hedef değişken string ise LabelEncoder uygulanır.
        """
        print("=== SKLEARN PIPELINE İLE MODEL EĞİTİMİ BAŞLADI ===")

        # Hedef tipini belirle ve gerekirse encode et
        y_encoded = y
        if model_type == 'auto':
            if y.dtype in ['object', 'bool'] or len(pd.Series(y).unique()) < 20:
                model_type = 'classification'
            else:
                model_type = 'regression'

        if model_type == 'classification' and y.dtype == 'object':
            self.target_label_encoder = LabelEncoder()
            y_encoded = self.target_label_encoder.fit_transform(y)
            print("Hedef değişken LabelEncoder ile dönüştürüldü.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )

        preprocessor = self._build_sklearn_preprocessing(X)

        if model_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, confusion_matrix
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            model = RandomForestRegressor(n_estimators=200, random_state=random_state)

        clf = SkPipeline(steps=[('preprocess', preprocessor), ('model', model)])
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        if model_type == 'classification':
            from sklearn.metrics import classification_report, confusion_matrix
            print("\n=== SINIFLANDIRMA (Sklearn Pipeline) ===")
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix (Sklearn Pipeline)')
            plt.ylabel('Gerçek Değerler')
            plt.xlabel('Tahmin Edilen Değerler')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'confusion_matrix_sklearn_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Confusion matrix görseli kaydedildi: {os.path.abspath(plot_path)}")
        else:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            print("\n=== REGRESYON (Sklearn Pipeline) ===")
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
            plt.xlabel('Gerçek Değerler')
            plt.ylabel('Tahmin Edilen Değerler')
            plt.title('Gerçek vs Tahmin (Sklearn Pipeline)')
            plt.grid(True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'regression_actual_vs_pred_sklearn_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Regresyon görseli kaydedildi: {os.path.abspath(plot_path)}")

        self.pipeline_steps.append("Sklearn pipeline ile model eğitimi tamamlandı")
        return clf, X_train, X_test, y_train, y_test




