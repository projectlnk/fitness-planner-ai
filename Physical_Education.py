# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Getting path directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename):
    return os.path.join(CURRENT_DIR, filename)

st.set_page_config(
    page_title="AI Планировщик тренировок",
    page_icon="💪",
    layout="wide"
)

# Session_state intialization
if 'plan_generated' not in st.session_state:
    st.session_state['plan_generated'] = False
if 'progress_saved' not in st.session_state:
    st.session_state['progress_saved'] = False

st.title("💪 Алгоритмический планировщик физической нагрузки")
st.markdown("---")

@st.cache_data
def load_exercises():
    csv_path = get_data_path('exercises.csv')
    
    if not os.path.exists(csv_path):
        st.error(f"Файл exercises.csv не найден по пути: {csv_path}")
        st.info("Убедитесь, что файл exercises.csv находится в той же папке, что и программа")
        return pd.DataFrame()  # Returning empty DataFrame
    
    df = pd.read_csv(csv_path, encoding='utf-8', quoting=1)
    return df

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def create_embeddings(df, _model):
    # Forming text descriptors
    texts = []
    for _, row in df.iterrows():
        text = f"Упражнение: {row['name']}. Целевая группа мышц: {row['target_muscle']}. Уровень сложности: {row['difficulty']}. Инвентарь: {row['equipment']}. Описание: {row['description']}"
        texts.append(text)
    
    # Getting embeddings
    embeddings = _model.encode(texts, show_progress_bar=False)
    return embeddings, texts

@st.cache_data
def perform_clustering(embeddings, n_clusters=7):

    # Clustering
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_scaled)
    
    # PCA for visualisation
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings_scaled)
    
    return clusters, embeddings_2d, kmeans

df = load_exercises()

if df.empty:
    st.stop()

model = load_embedding_model()
embeddings, texts = create_embeddings(df, model)
clusters, embeddings_2d, kmeans = perform_clustering(embeddings)

st.sidebar.header("📊 Навигация")
page = st.sidebar.radio(
    "Выберите раздел",
    ["🎯 Планировщик тренировок", "🔬 Кластеризация упражнений", "📈 Мой прогресс"]
)

if page == "🎯 Планировщик тренировок":
    st.header("🎯 Персональный планировщик тренировок")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exercise_names = df['name'].tolist()
        target_exercise = st.selectbox("Выберите целевое упражнение", exercise_names)
        
        current_level = st.number_input(
            "Сколько повторений вы можете выполнить прямо сейчас?",
            min_value=0, max_value=30, value=0, step=1,
            help="Если не можете выполнить ни разу, поставьте 0"
        )
    
    with col2:
        all_equipment = set()
        for eq in df['equipment'].dropna():
            for e in str(eq).split('-'):
                all_equipment.add(e.strip())
        available_equipment = st.multiselect(
            "Какой инвентарь вам доступен?",
            sorted(list(all_equipment))
        )
        
        sessions_per_week = st.slider("Тренировок в неделю", 1, 5, 3)
    
    if st.button("🚀 Построить тренировочный план", type="primary"):
        st.session_state['plan_generated'] = True
        st.session_state['target_exercise'] = target_exercise
        st.session_state['current_level'] = current_level
        st.session_state['sessions_per_week'] = sessions_per_week
    
    if st.session_state.get('plan_generated', False):
        target_exercise = st.session_state['target_exercise']
        current_level = st.session_state['current_level']
        sessions_per_week = st.session_state['sessions_per_week']
        
        if current_level == 0:
            st.subheader("📋 Ваш план освоения упражнения")
            st.info(f"Вы не можете выполнить упражнение «{target_exercise}» ни разу. Мы построим путь от простого к сложному.")
            
            target_rows = df[df['name'] == target_exercise]
            if len(target_rows) > 0:
                target_row = target_rows.iloc[0]
                target_difficulty = target_row['difficulty']
                
                easier_exercises = df[df['difficulty'] < target_difficulty].head(8)
                
                if len(easier_exercises) > 0:
                    plan_weeks = []
                    week_num = 1
                    
                    for i, (_, ex_row) in enumerate(easier_exercises.iterrows()):
                        if i >= 6:
                            break
                        
                        reps_text = f"{5 + i*2} повторений" if ex_row['difficulty'] > 1 else "30 секунд (вис)"
                        week_plan = {
                            'week': week_num,
                            'exercise': ex_row['name'],
                            'sets': 3,
                            'reps': reps_text,
                            'rest': '60-90 секунд',
                            'progress_criteria': 'Выполнить 2 тренировки подряд без пропусков'
                        }
                        plan_weeks.append(week_plan)
                        week_num += 1
                    
                    plan_weeks.append({
                        'week': week_num,
                        'exercise': target_exercise,
                        'sets': 3,
                        'reps': 'Попытка выполнения 1-3 повторений',
                        'rest': '90-120 секунд',
                        'progress_criteria': 'Успешное выполнение с правильной техникой'
                    })
                    
                    for week in plan_weeks:
                        with st.expander(f"Неделя {week['week']}: {week['exercise']}"):
                            st.write(f"**Количество подходов:** {week['sets']}")
                            st.write(f"**Объём:** {week['reps']}")
                            st.write(f"**Отдых между подходами:** {week['rest']}")
                            st.write(f"**Критерий перехода:** {week['progress_criteria']}")
                else:
                    st.warning("Для этого упражнения пока нет подводящих упражнений в базе.")
            else:
                st.warning(f"Упражнение '{target_exercise}' не найдено в базе данных.")
        
        else:
            st.subheader("📋 Ваш план прогрессии")
            st.success(f"Ваш текущий уровень: {current_level} повторений. Цель: увеличить результат.")
            
            plan_weeks = []
            current_max = current_level
            
            for week in range(1, 7):
                if week > 1:
                    increase = max(1, int(current_max * 0.07))
                    current_max = min(current_max + increase, 25)
                
                week_plan = {
                    'week': week,
                    'week_start': (datetime.now() + timedelta(days=(week-1)*7)).strftime("%d.%m.%Y"),
                    'target_reps': current_max,
                    'plan': []
                }
                
                for day in range(1, sessions_per_week + 1):
                    if current_max >= 10:
                        reps_distribution = f"{current_max-2}, {current_max}, {current_max-1}"
                    elif current_max >= 5:
                        reps_distribution = f"{current_max-1}, {current_max}, {current_max-1}"
                    else:
                        reps_distribution = f"{current_max}, {current_max}, {current_max}"
                    
                    week_plan['plan'].append({
                        'day': day,
                        'exercise': target_exercise,
                        'sets': 3,
                        'reps': reps_distribution,
                        'rest': '90 секунд'
                    })
                
                plan_weeks.append(week_plan)
            
            for week in plan_weeks:
                with st.expander(f"Неделя {week['week']} (с {week['week_start']}) — цель: {week['target_reps']} повторений"):
                    progress_pct = min(100, int((week['target_reps'] / 25) * 100))
                    st.progress(progress_pct, text=f"Уровень сложности: {progress_pct}%")
                    
                    plan_df = pd.DataFrame(week['plan'])
                    plan_df = plan_df.rename(columns={
                        'day': 'День',
                        'exercise': 'Упражнение',
                        'sets': 'Подходы',
                        'reps': 'Повторения',
                        'rest': 'Отдых'
                    })
                    st.table(plan_df[['День', 'Упражнение', 'Подходы', 'Повторения', 'Отдых']])
                    
                    button_key = f"save_week_{week['week']}_{target_exercise.replace(' ', '_')}"
                    if st.button(f"✅ Отметить неделю {week['week']} как выполненную", key=button_key):
                        history_file = get_data_path('history.json')
                        if os.path.exists(history_file):
                            with open(history_file, 'r', encoding='utf-8') as f:
                                history = json.load(f)
                        else:
                            history = {}
                        
                        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        history[today] = {
                            'exercise': target_exercise,
                            'reps': week['target_reps'],
                            'week': week['week']
                        }
                        
                        with open(history_file, 'w', encoding='utf-8') as f:
                            json.dump(history, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"✅ Прогресс сохранён! Неделя {week['week']} отмечена. Перейдите во вкладку «Мой прогресс».")
                        st.balloons()
                        
                        st.session_state['progress_saved'] = True
        
        st.info("💡 Совет: после выполнения недели нажимайте кнопку «Отметить неделю» — данные сохранятся в историю прогресса.")

elif page == "🔬 Кластеризация упражнений":
    st.header("🔬 Автоматическая кластеризация упражнений")
    st.markdown("Упражнения сгруппированы на основе **нейросетевых векторных представлений** (sentence-transformers).")
    
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    cluster_descriptions = {}
    for c in range(max(clusters) + 1):
        cluster_exercises = df_with_clusters[df_with_clusters['cluster'] == c]['name'].tolist()
        main_muscles = df_with_clusters[df_with_clusters['cluster'] == c]['target_muscle'].mode()
        avg_diff = df_with_clusters[df_with_clusters['cluster'] == c]['difficulty'].mean()
        
        cluster_descriptions[c] = {
            'exercises': cluster_exercises[:5],
            'main_muscle': main_muscles[0] if len(main_muscles) > 0 else "смешанная",
            'avg_difficulty': round(avg_diff, 1)
        }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Визуализация кластеров (PCA)")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=clusters, cmap='tab10', alpha=0.7, s=100)
        
        for i, idx in enumerate(range(0, len(df), 5)):
            if idx < len(df):
                ax.annotate(df.iloc[idx]['name'][:15], 
                           (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel("Первая главная компонента")
        ax.set_ylabel("Вторая главная компонента")
        ax.set_title("Карта упражнений: цвет = кластер")
        plt.colorbar(scatter, label='Кластер')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Характеристики кластеров")
        for c in range(max(clusters) + 1):
            with st.expander(f"Кластер {c+1}: {cluster_descriptions[c]['main_muscle']} (сложность {cluster_descriptions[c]['avg_difficulty']})"):
                st.write("**Примеры упражнений:**")
                for ex in cluster_descriptions[c]['exercises']:
                    st.write(f"• {ex}")
    
    st.subheader("Полный список упражнений с принадлежностью к кластерам")
    display_df = df_with_clusters[['name', 'target_muscle', 'difficulty', 'equipment', 'cluster']]
    display_df = display_df.rename(columns={
        'name': 'Упражнение',
        'target_muscle': 'Группа мышц',
        'difficulty': 'Сложность',
        'equipment': 'Инвентарь',
        'cluster': 'Кластер'
    })
    st.dataframe(display_df, use_container_width=True)

else:
    st.header("📈 Мой прогресс")
    
    history_file = get_data_path('history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {}
    
    if len(history) > 0:
        history_data = []
        for k, v in history.items():
            history_data.append({
                'date': k, 
                'reps': v['reps'],
                'exercise': v.get('exercise', 'неизвестно'),
                'week': v.get('week', '?')
            })
        
        history_df = pd.DataFrame(history_data)
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.sort_values('date')
        
        st.subheader("Динамика результатов")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history_df['date'], history_df['reps'], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel("Дата")
        ax.set_ylabel("Максимальное количество повторений")
        ax.set_title("Ваш прогресс по выбранному упражнению")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("История тренировок")
        display_history = history_df.rename(columns={
            'date': 'Дата', 
            'reps': 'Повторения',
            'exercise': 'Упражнение',
            'week': 'Неделя'
        })
        st.dataframe(display_history[['Дата', 'Упражнение', 'Неделя', 'Повторения']], use_container_width=True)
        
        if st.button("🗑 Очистить историю"):
            os.remove(history_file)
            st.success("История очищена!")
            st.rerun()
    else:
        st.info("📭 История тренировок пуста. Начните выполнять план из раздела «Планировщик тренировок» и отмечайте успехи.")
        st.markdown("""
        **Как заполнить историю?**
        1. Перейдите во вкладку «Планировщик тренировок»
        2. Выберите целевое упражнение (например, «Отжимания от пола»)
        3. Укажите текущий уровень (например, 5 повторений)
        4. Нажмите «Построить тренировочный план»
        5. После выполнения недели нажмите кнопку «Отметить неделю как выполненную»
        """)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **О системе**
    
    Данный планировщик использует:
    - 🧠 Нейросетевые эмбеддинги упражнений
    - 🔬 Кластеризацию KMeans (7 кластеров)
    - 📅 Привязку планов ко времени
    - 📊 Отслеживание прогресса
    
    **Преимущество:** безопасные планы на основе проверенных источников
    """
)

st.sidebar.markdown("---")
st.sidebar.caption(f"База упражнений: {len(df)} движений")
st.sidebar.caption(f"Выявлено кластеров: {max(clusters) + 1}")