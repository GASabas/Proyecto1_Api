import pandas as pd
import numpy as np
from fastapi import FastAPI

app = FastAPI()


movies_credits= pd.read_csv("./Movies/movies_credits.csv")
movies_credits['release_date'] = pd.to_datetime(movies_credits['release_date'], errors='coerce')
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
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    
    actor_data = movies_credits[movies_credits['cast'].apply(lambda x: any(actor['name'] == nombre_actor for actor in x))]
    
    
    actor_data = actor_data[actor_data['director'] != nombre_actor]

    
    if actor_data.empty:
        return {"error": f"No se encontró ningún actor con el nombre '{nombre_actor}'."}

    
    cantidad_peliculas = len(actor_data)

    
    actor_data['return'] = actor_data.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0, axis=1)
    retorno_total = actor_data['return'].sum()
    promedio_retorno = actor_data['return'].mean()

    
    return {
        "message": f"El actor '{nombre_actor}' ha participado en {cantidad_peliculas} cantidad de filmaciones, "
                   f"el mismo ha conseguido un retorno de {retorno_total} con un promedio de {promedio_retorno} por filmación."
    }
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    
    actor_data = movies_credits[movies_credits['actor'] == nombre_actor]
    
    
    if actor_data.empty:
        return {"error": f"No se encontró ningún actor con el nombre '{nombre_actor}'."}
    
    
    actor_data['return'] = actor_data.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0, axis=1)
    retorno_total = actor_data['return'].sum()
    promedio_retorno = actor_data['return'].mean()
    
    
    peliculas_info = actor_data[['title', 'release_date', 'return', 'budget', 'revenue']]
    peliculas_info = peliculas_info.rename(columns={
        'title': 'Película',
        'release_date': 'Fecha de Lanzamiento',
        'return': 'Retorno',
        'budget': 'Costo',
        'revenue': 'Ganancia'
    })
    
    
    peliculas_detalle = peliculas_info.to_dict(orient='records')
    
    return {
        "message": f"El actor '{nombre_actor}' ha participado en {len(actor_data)} películas, ha conseguido un retorno total de {retorno_total} con un promedio de {promedio_retorno}.",
        "peliculas": peliculas_detalle
    }
