

from health_data_pipeline import HealthDataPipeline
import warnings

warnings.filterwarnings('ignore')


def main():
    """Ana fonksiyon - Pipeline'ı çalıştır"""

    print("🏥 SAĞLIK VERİSİ PIPELINE ÖRNEĞİ")
    print("=" * 50)

    # 1. Pipeline'ı başlat
    print("\n1. Pipeline başlatılıyor...")
    pipeline = HealthDataPipeline()

    # 2. Veri yükleme
    print("\n2. Veri yükleniyor...")
    data = pipeline.load_data('Talent_Academy_Case_DT_2025.xlsx')



    print(f"✅ Veri başarıyla yüklendi. Boyut: {data.shape}")

    # 3. Veri keşfi
    print("\n3. Veri keşfi yapılıyor...")
    pipeline.explore_data()

    # 4. Veri temizliği
    print("\n4. Veri temizliği yapılıyor...")
    pipeline.clean_data()

    # 5. Özellik mühendisliği
    print("\n5. Özellik mühendisliği yapılıyor...")
    pipeline.feature_engineering()

    # 6. Özellik ölçeklendirme
    print("\n6. Özellik ölçeklendirme yapılıyor...")
    pipeline.scale_features()

    # 7. Modelleme hazırlığı
    print("\n7. Modelleme hazırlığı yapılıyor...")
    X, y = pipeline.prepare_for_modeling(target_column='Yas')

    if X is None or y is None:
        print("❌ Modelleme hazırlığı başarısız!")
        return

    print(f" Özellik matrisi: {X.shape}")
    print(f" Hedef değişken: {y.shape}")

    # 8. Model eğitimi
    print("\n8. Model eğitimi yapılıyor...")
    model, X_train, X_test, y_train, y_test = pipeline.train_model(X, y, model_type='auto')

    # 9. Pipeline özeti
    print("\n9. Pipeline özeti alınıyor...")
    summary = pipeline.get_pipeline_summary()

    print("\n" + "=" * 50)
    print("🎉 PIPELINE BAŞARIYLA TAMAMLANDI!")
    print("=" * 50)
    print(f"Toplam adım sayısı: {summary['total_steps']}")
    print(f"Ham veri boyutu: {summary['raw_data_shape']}")
    print(f"İşlenmiş veri boyutu: {summary['processed_data_shape']}")

    print("\nPipeline adımları:")
    for i, step in enumerate(summary['pipeline_steps'], 1):
        print(f"  {i}. {step}")

    # 10. Pipeline'ı kaydet
    print("\n10. Pipeline kaydediliyor...")
    pipeline.save_pipeline('health_pipeline.joblib')

    print("\n✅ Pipeline başarıyla kaydedildi: 'health_pipeline.joblib'")

    return pipeline, model


def run_full_pipeline_example():
    """Tek seferde tüm pipeline'ı çalıştır"""

    print("\n" + "🚀" * 20)
    print("TEK SEFERDE TAM PIPELINE ÇALIŞTIRILIYOR")
    print("🚀" * 20)

    # Yeni pipeline oluştur ve veri dosyası yolunu belirt
    full_pipeline = HealthDataPipeline('Talent_Academy_Case_DT_2025.xlsx')

    # Tüm adımları tek seferde çalıştır (EDA dahil)
    model, X_train, X_test, y_train, y_test = full_pipeline.run_full_pipeline(
        target_column='Yas',
        save_pipeline=True,
        pipeline_path='full_health_pipeline.joblib',
        include_eda=True
    )

    if model is not None:
        print("\n✅ Tam pipeline başarıyla tamamlandı!")
        return full_pipeline, model
    else:
        print("\n❌ Tam pipeline başarısız!")
        return None, None


def run_classification_example():
    """Kategorik hedef ile sınıflandırma örneği (Confusion Matrix üretir)"""

    print("\n" + "📊" * 20)
    print("CİNSİYET HEDEFİ İLE SINIFLANDIRMA ÖRNEĞİ")
    print("📊" * 20)

    # Yeni pipeline oluştur
    pipeline = HealthDataPipeline()

    # Veri yükle
    data = pipeline.load_data('Talent_Academy_Case_DT_2025.xlsx')
    if data is None:
        print("❌ Veri yüklenemedi! Dosya yolunu kontrol edin.")
        return None, None

    # Temizlik ve dönüşümler
    pipeline.clean_data()
    # Hedef sütunu one-hot'tan koru ki prepare_for_modeling hedefi bulabilsin
    pipeline.feature_engineering(preserve_columns=['Cinsiyet'])
    pipeline.scale_features()

    # Modelleme hazırlığı (kategorik hedef: Cinsiyet)
    X, y = pipeline.prepare_for_modeling(target_column='Cinsiyet')
    if X is None or y is None:
        print("❌ Modelleme hazırlığı başarısız!")
        return None, None

    # Eğitim (sınıflandırma modu)
    model, X_train, X_test, y_train, y_test = pipeline.train_model(
        X, y, model_type='classification'
    )

    print("\n✅ Sınıflandırma tamamlandı. Confusion Matrix outputs/ klasörüne kaydedildi.")
    return pipeline, model


def load_saved_pipeline_example():
    """Kaydedilmiş pipeline'ı yükle"""

    print("\n" + "📂" * 20)
    print("KAYDEDİLMİŞ PIPELINE YÜKLENİYOR")
    print("📂" * 20)

    try:
        # Yeni pipeline oluştur
        loaded_pipeline = HealthDataPipeline()

        # Kaydedilmiş pipeline'ı yükle
        loaded_pipeline.load_pipeline('full_health_pipeline.joblib')

        print("✅ Pipeline başarıyla yüklendi!")
        print(f"Pipeline adımları: {loaded_pipeline.pipeline_steps}")

        return loaded_pipeline

    except FileNotFoundError:
        print("❌ Pipeline dosyası bulunamadı! Önce pipeline'ı çalıştırın.")
        return None


if __name__ == "__main__":
    # Ana pipeline'ı çalıştır
    pipeline, model = main()

    # Tam pipeline örneği
    full_pipeline, full_model = run_full_pipeline_example()

    # Sınıflandırma örneği (Cinsiyet hedefi) - Confusion Matrix üretir
    clf_pipeline, clf_model = run_classification_example()

    # Kaydedilmiş pipeline'ı yükle
    loaded_pipeline = load_saved_pipeline_example()

    print("\n" + "🎯" * 30)
    print("TÜM ÖRNEKLER TAMAMLANDI!")
    print("🎯" * 30)
