import streamlit as st
import pickle
import numpy as np
import json
# uzgarish
# Modelni yuklash
with open('aholimodel.pkl', 'rb') as f:
    model = pickle.load(f)

# Mamlakatlar ro'yxatini JSON dan o'qish
with open('country_list.json', 'r') as file:
    countries = json.load(file)

# Mamlakatlar nomlari ro'yxati
country_names = [country['Country Name'] for country in countries]

# Streamlit interfeysi
st.title("Aholi O'sishini Bashorat Qilish Ilovasi")
st.write("Mamlakat va yillar bo'yicha aholi sonini bashorat qiling!")

# Mamlakat va yilni tanlash
country = st.selectbox('Mamlakatni tanlang:', country_names)
year = st.number_input('Bashorat qilish yili:', min_value=2025, max_value=2040, value=2026)

# Bashorat qilish tugmasi
if st.button('Bashorat Qiling'):
    prediction = model.predict(np.array([[year]]))[0]  # Bashorat qilish
    st.success(f"{country} uchun {year}-yilga aholi soni bashorati: {int(prediction)}")

#uzgarish


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import json

# # Datasetni yuklash
# df = pd.read_csv('pop.csv')

# # Funksiya: Mamlakat ro'yxatini yaratish
# def country_list_gen(df):
#     df.rename(columns={'Country Name': 'country_name'}, inplace=True)
#     df['country_name'] = df['country_name'].apply(lambda row: row.lower())
#     lists = df['country_name'].unique().tolist()
#     with open('country_list.json', 'w', encoding='utf-8') as f:
#         json.dump(lists, f, ensure_ascii=False, indent=4)
#     return lists, df

# # Funksiya: Tanlangan mamlakat ma'lumotlarini olish
# def selecting_country(df, country):
#     df = df.loc[df['country_name'] == country]
#     df.drop(['country_name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
#     df = df.T
#     df.dropna(inplace=True)
#     df.reset_index(inplace=True)
#     df.columns = ['Year', 'Population']
#     df['Year'] = df['Year'].astype(int)
#     return df

# # Funksiya: Model yaratish
# def prediction_model(df):
#     x = df['Year'].values.reshape(-1, 1)
#     y = df['Population'].values.reshape(-1, 1)
#     model = LinearRegression().fit(x, y)
#     return model

# # Funksiya: Bashorat qilish
# def prediction(model, year):
#     return int(model.coef_[0][0] * year + model.intercept_[0])

# # Streamlit interfeysi
# st.title("Aholi O'sishini Bashorat Qilish Ilovasi")
# st.write("Mamlakat va yillar bo'yicha aholi sonini bashorat qiling!")

# # Mamlakatni tanlash
# lists, df = country_list_gen(df)
# selected_country = st.selectbox("Mamlakatni tanlang:", sorted(lists))

# # Bashorat qilish uchun yil kiritish
# year = st.number_input("Bashorat qilish yili:", min_value=2025, max_value=2100, value=2025)

# # Bashorat tugmasi
# if st.button("Bashorat Qiling"):
#     country_df = selecting_country(df, selected_country)
#     model = prediction_model(country_df)
#     result = prediction(model, year)
    
#     # Natijalarni ko'rsatish
#     st.success(f"{selected_country.upper()} mamlakati uchun {year}-yildagi bashorat qilingan aholi soni: {result:,d}")








