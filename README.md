Proyecto Integrador
El proyecto consta de 6 archivos que están en el proyecto de github, donde voy a desarrollar 2 archivos específicos que son PI1_SABAS.ipynb y por otro lado PI_FastApi_Sabas.ipynb. Además voy a mencionar de donde provienen los archivos main.py y movies_credits.csv, que son necesarios para el desarrollo de la Api.
A fines prácticos los desarrollos se harán por ítems con numeración.
•	Archivo PI1_SABAS.ipynb
Archivo movies_datasets.csv
1.	Se importan las librerías que a priori serán necesarias, Pandas y Numpy.
2.	Se asigna el nombre movies al leer el archivo movies_dataset.csv
3.	Verifico como están designadas las columnas del dataframe
4.	Después verifico la columna anidada belongs_to_collection y el tipo de dato que es. 
5.	Limpio y estandarizo el dato “name”, y trato de evitar errores
6.	Extraigo el nombre
7.	Luego extraigo el id de la película
8.	Sigo con los géneros de las películas donde hago una fila por cada genero que contiene la película, manteniendo los demás datos.
9.	Después hago lo mismo con las compañías productoras de cada película
10.	Siguiendo de igual manera con los lenguajes
11.	Luego de desanidar las columnas y extraer los datos que creo convenientes, elimino las columnas que no serán necesarias.
12.	Reemplazo los valores NaN por 0, en revenue y Budget
13.	Extraigo la fecha de lanzamiento o de estreno de las películas de su fecha completa proveniente de formato año-mes-día
14.	Calculo el retorno de las películas
Archivo credits.csv
1.	Abro el archivo csv
2.	Procedo a extraer el Nombre del actor principal desde la columna cast
3.	Extraigo al director de la columna crew
4.	Elimino las columnas cast y crew, resultando un data frame con 3 columnas (Id,actor principal y director)
5.	Procedo a unir los dos archivos resultantes de movies_dataset y credits en un solo dataframe
Archivo movies_credits.csv
1.	Hago una selección para disminuir la cantidad de datos del dataframe, donde hago que seleccione según el promedio de los votos sea mayor a 6, y además donde su fecha de lanzamiento sea después de los 1980.
2.	Luego reduzco la cantidad de datos a 10000 filas y lo convierto al csv previamente nombrado


•	Archivo PI1_FastApi_Sabas.ipynb
Este nuevo archivo es para que sea mas fácil de utilizar en render para poder crear la api.
1.	Abro el archivo creado previamente llamado movies_credits.csv
2.	Creo la función para obtener la cantidad de películas estrenadas durante un mes específico del año
3.	Luego lo mismo para la cantidad de películas estrenadas en un dia especifico de la semana
4.	La puntuación de la película según la popularidad, con el año de estreno
5.	La cantidad de votos con a la puntuación promedio obtenida
6.	Luego definimos la película en la que actuó el actor, en mi caso seria el actor principal de la película, con el respectivo retorno y el promedio de los retornos de las películas en las que ha actuado
7.	Lo mismo se hizo para el director, para las películas que dirigió con su respectivo retorno
8.	Después se hizo para la recomendación de película según su genero 
9.	Se lo convierte en py y se lo renombra como main.py para que sea mejor absorbido por render.
