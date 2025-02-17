#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()



# In[13]:


movies_credits= pd.read_csv("movies_credits.csv")


# In[14]:


@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5,
        'junio': 6, 'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10,
        'noviembre': 11, 'diciembre': 12
    }

    
    if mes.lower() not in meses:
        return {"error": "Mes inválido. Debe ser uno de los siguientes: enero, febrero, marzo, etc."}

    
    mes_numero = meses[mes.lower()]
    peliculas_mes = movies_credits[movies_credits['release_date'].dt.month == mes_numero]

    
    cantidad = len(peliculas_mes)
    return {"message": f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}"}


# In[15]:


@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    
    dias = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3, 'viernes': 4,
        'sábado': 5, 'domingo': 6
    }

    
    if dia.lower() not in dias:
        return {"error": "Día inválido. Debe ser uno de los siguientes: lunes, martes, miércoles, jueves, viernes, sábado, domingo."}

    
    dia_numero = dias[dia.lower()]
    peliculas_dia = movies_credits[movies_credits['release_date'].dt.dayofweek == dia_numero]

    
    cantidad = len(peliculas_dia)
    return {"message": f"{cantidad} cantidad de películas fueron estrenadas en los días {dia.capitalize()}"}


# In[16]:


@app.get("/score_titulo/{titulo_de_la_filmacion}")
def score_titulo(titulo_de_la_filmacion: str):
    
    pelicula = movies_credits[movies_credits['title'].str.contains(titulo_de_la_filmacion, case=False, na=False)]
    
    
    if pelicula.empty:
        return {"error": f"No se encontró ninguna película con el título '{titulo_de_la_filmacion}'."}
    
   
    titulo = pelicula['title'].iloc[0]
    año_estreno = pelicula['release_year'].iloc[0]
    score = pelicula['vote_average'].iloc[0]
    
    
    return {
        "message": f"La película '{titulo}' fue estrenada en el año {año_estreno} con un score/popularidad de {score}."}


# In[17]:


@app.get("/votos_titulo/{titulo_de_la_filmacion}")
def votos_titulo(titulo_de_la_filmacion: str):
    
    pelicula = movies_credits[movies_credits['title'].str.contains(titulo_de_la_filmacion, case=False, na=False)]
    
    
    if pelicula.empty:
        return {"error": f"No se encontró ninguna película con el título '{titulo_de_la_filmacion}'."}
    
    
    cantidad_votos = pelicula['vote_count'].iloc[0]
    promedio_votacion = pelicula['vote_average'].iloc[0]
    
    
    if cantidad_votos < 2000:
        return {"message": f"La película '{titulo_de_la_filmacion}' no tiene suficientes valoraciones. Solo tiene {cantidad_votos} votos."}
    
    
    titulo = pelicula['title'].iloc[0]
    año_estreno = pelicula['release_year'].iloc[0]
    
    
    return {
        "message": f"La película '{titulo}' fue estrenada en el año {año_estreno}. La misma cuenta con un total de {cantidad_votos} valoraciones, con un promedio de {promedio_votacion}."
    }


# In[ ]:


@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    actor_data = movies_credits[movies_credits['principal_actor'] == nombre_actor]

    if actor_data.empty:
        return {"error": f"No se encontró ningún actor con el nombre '{nombre_actor}'."}

    # Mantener solo una fila por película, priorizando la de mayor revenue
    actor_data = actor_data.sort_values(by='revenue', ascending=False).drop_duplicates(subset=['title'])

    actor_data = actor_data.copy()
    actor_data['return'] = actor_data.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0, axis=1)
    retorno_total = round(actor_data['return'].sum(), 2)
    promedio_retorno = round(actor_data['return'].mean(), 2)  

    peliculas_info = actor_data[['title', 'release_date', 'return', 'budget', 'revenue']].rename(columns={
        'title': 'Película',
        'release_date': 'Fecha de Lanzamiento',
        'return': 'Retorno',
        'budget': 'Costo',
        'revenue': 'Ganancia'
    })

    peliculas_detalle = peliculas_info.to_dict(orient='records')

    return {
        "message": f"El actor '{nombre_actor}' ha participado en {len(actor_data)} películas, consiguiendo un retorno total de {retorno_total} con un promedio de {promedio_retorno}.",
        "peliculas": peliculas_detalle
    }


# In[ ]:


@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    director_data = movies_credits[movies_credits['director'] == nombre_director]

    if director_data.empty:
        return {"error": f"No se encontró ningún director con el nombre '{nombre_director}'."}

    
    director_data = director_data.sort_values(by='revenue', ascending=False).drop_duplicates(subset=['title'])

    director_data = director_data.copy()
    director_data['return'] = director_data.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0, axis=1)

    retorno_total = round(director_data['return'].sum(), 2)
    promedio_retorno = round(director_data['return'].mean(), 2) 

    peliculas_info = director_data[['title', 'release_date', 'return', 'budget', 'revenue']].rename(columns={
        'title': 'Película',
        'release_date': 'Fecha de Lanzamiento',
        'return': 'Retorno',
        'budget': 'Costo',
        'revenue': 'Ganancia'
    })

    peliculas_detalle = peliculas_info.to_dict(orient='records')

    return {
        "message": f"El director '{nombre_director}' ha dirigido {len(director_data)} películas, consiguiendo un retorno total de {retorno_total} con un promedio de {promedio_retorno} por filmación.",
        "peliculas": peliculas_detalle
    }


# In[20]:


movies_credits = movies_credits.dropna(subset=['genre_name'])  # Eliminar NaN
movies_credits = movies_credits[movies_credits['genre_name'].str.strip() != ""]  # Eliminar valores vacíos
movies_credits['genre_name'] = movies_credits['genre_name'].astype(str).str.lower().str.strip()  # Convertir a string

movies_unique = movies_credits.drop_duplicates(subset=['genre_name'])

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_unique['genre_name'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movies_unique['title'] = movies_unique['title'].astype(str).str.lower().str.strip()
indices = pd.Series(movies_unique.index, index=movies_unique['title'])

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    try:
        titulo_normalizado = titulo.strip().lower()

        
        idx = indices.get(titulo_normalizado)
        if idx is None:
            return {"error": "Película no encontrada"}

        
        if isinstance(idx, (pd.Series, np.ndarray, list)):
            idx = idx.iloc[0] if isinstance(idx, pd.Series) else idx[0]

        
        sim_scores = list(enumerate(cosine_sim[int(idx)]))  
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6] 
        sim_scores = [x for x in sim_scores if x[0] != idx][:5] 
        movie_indices = [i[0] for i in sim_scores]

        recommended_movies = movies_unique['title'].iloc[movie_indices].tolist()
        return {"películas recomendadas": recommended_movies}

    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}


# In[21]:


import nbformat
from nbconvert import PythonExporter

with open("PI1_FastApi_Sabas.ipynb") as f:
    notebook = nbformat.read(f, as_version=4)

exporter = PythonExporter()
script, _ = exporter.from_notebook_node(notebook)

with open("PI1_FastApi_Sabas.py", "w") as f:
    f.write(script)

