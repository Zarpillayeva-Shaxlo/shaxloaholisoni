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



import streamlit as st
import pandas as pd
import joblib  # Model yuklash uchun
import json

# Modelni yuklash
model = joblib.load("aholimodel3.pkl")  # Model fayli yuklanadi

# Datasetni yuklash
df = pd.read_csv('pop.csv')

# Funksiya: Mamlakat ro'yxatini yaratish
def country_list_gen(df):
    df.rename(columns={'Country Name': 'country_name'}, inplace=True)
    df['country_name'] = df['country_name'].apply(lambda row: row.lower())
    lists = df['country_name'].unique().tolist()
    with open('country_list.json', 'w', encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False, indent=4)
    return lists, df

# Funksiya: Tanlangan mamlakat ma'lumotlarini olish
def selecting_country(df, country):
    df = df.loc[df['country_name'] == country]
    df.drop(['country_name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
    df = df.T
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df.columns = ['Year', 'Population']
    df['Year'] = df['Year'].astype(int)
    return df

# Streamlit interfeysi
st.title("Aholi O'sishini Bashorat Qilish Ilovasi")
st.write("Mamlakatni tanlang va bashorat qiling!")

# Mamlakatni tanlash
lists, df = country_list_gen(df)
selected_country = st.selectbox("Mamlakatni tanlang:", sorted(lists))

# Bashorat qilish uchun yil kiritish
year = st.number_input("Bashorat qilish yili:", min_value=2025, max_value=2100, value=2025)

# Bashorat tugmasi
if st.button("Bashorat Qiling"):
    country_df = selecting_country(df, selected_country)
    
    # Input ma'lumotlarni tayyorlash
    country_df = country_df.tail(1)  # Eng so‘nggi ma'lumotlarni olish
    input_data = pd.DataFrame({
        'Year': [year]
    })
    
    # Model yordamida bashorat qilish
    result = model.predict(input_data)[0]  # `model.pkl` orqali bashorat
    
    # Natijalarni ko'rsatish
    st.success(f"{selected_country.upper()} mamlakati uchun {year}-yildagi bashorat qilingan aholi soni: {int(result):,d}")





