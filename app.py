import pandas as pd 
from sklearn.linear_model import LogisticRegression
import streamlit as st


data = pd.DataFrame({
    "math": [1, 0, 0, 0, 1, 1, 0, 0],
    "biology": [0, 1, 0, 0, 0, 0, 1, 0],
    "communication": [0, 1, 1, 0, 1, 0, 0, 1],
    "creativity": [0, 0, 0, 1, 0, 1, 0, 1],
    "field": ["IT", "Медицина", "Humanities", "Creative", "Business", "IT", "Медицина", "Humanities"]
})


X = data[["math", "biology", "communication", "creativity"]]
Y = data["field"]


model = LogisticRegression()
model.fit(X, Y)


st.title('ИИ-помощник в выборе профессии')


st.write('Ответь на вопросы:')


math = st.checkbox('Люблю математику')
bio = st.checkbox('Люблю биологию')
comm = st.checkbox('Люблю общение')
crea = st.checkbox('Люблю творчество')


if st.button('Получить рекомендацию'):
    prediction = model.predict([[int(math), int(bio), int(comm), int(crea)]])
    st.success(f'Рекомендуемое направление: {prediction[0]}') 
