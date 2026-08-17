import streamlit as st
import pandas as pd
from PIL import Image
from transformers import pipeline

# 1. Настройка внешнего вида сайта
st.set_page_config(page_title="Скупка-Сканер Excel", page_icon="📊")
st.title("📊 Сканер скупки с базой в Excel")
st.write("Сделайте фото, и приложение найдет цену в вашей таблице.")

# 2. Загрузка ИИ-моделей (загружаются один раз и работают бесплатно)
@st.cache_resource
def load_models():
    # Модель распознавания объектов от Google
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    # Модель автоматического перевода на русский язык
    translator = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")
    return classifier, translator

classifier, translator = load_models()

# 3. Функция поиска совпадений в Excel-таблице
def find_price_in_excel(detected_text):
    try:
        # Читаем таблицу Excel из папки с проектом
        df = pd.read_excel("prices.xlsx")
        
        # Переводим буквы в нижний регистр, чтобы поиск не ошибался из-за больших букв
        df['Предмет_low'] = df['Предмет'].astype(str).str.lower()
        search_query = detected_text.lower()
        
        # Ищем совпадение текста ИИ с текстом в таблице
        for _, row in df.iterrows():
            item_in_table = row['Предмет_low']
            if item_in_table in search_query or search_query in item_in_table:
                return row['Предмет'], row['Цена']
        return None, None
    except Exception as e:
        st.error(f"Не удалось прочитать файл prices.xlsx. Проверьте, загружен ли он. Ошибка: {e}")
        return None, None

# 4. Выбор источника фотографии
source_photo = st.radio("Источник фото:", ("Камера смартфона/ПК", "Галерея"))

if source_photo == "Камера смартфона/ПК":
    img_file = st.camera_input("Сделайте снимок предмета")
else:
    img_file = st.file_uploader("Выберите картинку из галереи...", type=["jpg", "jpeg", "png"])

# 5. Главная логика обработки картинки
if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Загруженное фото", use_container_width=True)
    
    st.write("⏳ Распознаю предмет и сверяю с Excel...")
    
    # Нейросеть распознает объект (выдает ответ на английском)
    predictions = classifier(image)
    best_match_en = predictions['label']
    
    # Очищаем текст от лишних запятых, если ИИ выдал несколько названий
    clean_en = best_match_en.split(',')[0]
    
    # Переводим английское название на русский язык
    translation_result = translator(clean_en)
    best_match_ru = translation_result[0]['translation_text']
    
    st.markdown(f"🤖 **ИИ считает, что на фото:** `{best_match_ru}` *(на английском: {clean_en})*")
    
    # Сначала ищем в Excel по русскому названию
    matched_name, price_value = find_price_in_excel(best_match_ru)
    
    # Если на русском не нашли, проверяем на всякий случай по английскому оригиналу
    if not price_value:
        matched_name, price_value = find_price_in_excel(clean_en)
        
    # Выводим итоговый результат на экран
    if price_value:
        st.success(f"✅ **Найдено совпадение в прайсе:** {matched_name}")
        st.metric(label="💰 ЦЕНА СКУПКИ", value=str(price_value))
    else:
        st.warning(f"⚠️ В вашей таблице `prices.xlsx` пока нет цены для '{best_match_ru}'.")
        st.info("Вы можете открыть свой Excel, добавить туда строку с этим предметом, и цена появится!")
