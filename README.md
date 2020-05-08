# Generando Musica con LSTM
#### 5/8/2020

## Una breve introduccion 
<div style="text-align: justify"> 
Buscando proyectos sobre aplicaciones de redes neuronales recurrentes, El principal tema que me llamaba la atencion era sobre generación de música y tenia tiempo queriendo realizar un proyecto similar mas que nada porque los modelos que pueden utilizarse son muy interesantes y divertidas de poner en practica. Busque unos proyectos relacionados a esto. A los inicios del proyecto mi idea era generar musica de Star Wars asi que empeze mi busqueda por el sagrado y vasto mundo de google como todo buen programador, encontre unos buenos e interesantes proyectos: uno utilizaba un modelo muy complejo que se basaba en un articulo publicado por un estudiante el modelo se llamaba C-RNN-GAN (Continuous Recurrent Neural Network Generative Adversial Model) esta sopa de letras pueden verla <a href="https://github.com/olofmogren/c-rnn-gan">aqui</a> este blog y repositorio fueron hechos por <a href="https://github.com/olofmogren">olofmorgen</a> les recomiendo checar su github y ver sus proyectos son muy interesantes, pero al intentar estudiar el modelo y entenderle y con apoyo de mis maestros me di cuenta que este modelo aunque interesante era muy complicado para el tiempo que tenia asi que decidi buscar otro y al final me encontre con un modelo llamado LSTM en un post el cual generaba musica de videojuegos lo lei y me parecio bastante interesante y suficientemente sencillo como para intentar replicarlo, pero antes debemos entender que es una RNN o Red Neuronal Recurrente y que significa LSTM, depues de esta no tan breve introduccion, es momento de aprender por encima que es una red recurrente y que rayos es un LSTM:
</div>

## Redes Neuronales Recurrentes
<div style="text-align: justify"> 
En resumen son redes que contienen un ciclo y permiten que informacion persista, Los humanos no comienzan a pensar desde cero cada segundo. A medida que lees este blog, comprendes cada palabra en base a tu conocimiento de las palabras previas. No tiras todo y comienzas a pensar desde cero de nuevo. Tus pensamientos tienen persistencia.<br><br> redes tradicionales no tienen este beneficio lo cual podria considerarse un gran defecto, por ejemplo digamos que ves una pelicula y quieres clasificar cada punto de la misma una red neuronal no podria razonar puntos previos de la pelicula para informar de los nuevos puntos.
</div>

<p align="center">
  <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png">
</p>


<p align="center">
  <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png">
</p>

<div style="text-align: justify"> 
</div>

## LSTM (Long Short Term Memory)
<div style="text-align: justify">
Las redes LSTM son una variante de red neuronal recurrente. Estas redes surgieron principalmente para solventar un problema de dependencia de las redes recurrentes. Cuando el contexto necesario para predecir es de relativamente corto plazo, las redes recurrentes no tienen problemas, pero cuando se necesita de información que se remonte a entradas mucho más anteriores, se les dificulta predecir correctamente. No es que sea imposible, es técnicamente cierto que bajo ciertos pesos es posible que la red no tenga problemas de ese tipo, pero en la práctica esto no sucede. Aquí es donde entran las redes LSTM a salvar el día ya que fueron diseñadas para combatir justo ese problema.<br><br> Así como las redes recurrentes, las redes LSTM no son especialmenten nuevas. Existen desde más o menos 1990 pero han tenido que esperar hasta el poder de computo actual para poder mostrar buenos resultados.
</div>

<p align="center">
  <img src="https://i.pinimg.com/originals/27/95/bc/2795bc16b012322f7767cd4d940ba2e3.png">
</p>

## Music21
<div style="text-align: justify">
Lo que hace el proyecto es leer la música en formato MIDI usando music21 para extraer las notas de lo archivos MIDI con los que se quiere aprender para generar música similar.<br><br> Music21 es una coleccion de herramientas para ayudar a estudiantes y otros hallar respuestas de musica mas facilmente, cosas como "Me pregunto cuantas veces Bach hace eso" o "Desearia conocer cual banda fue la primera en usar ese progeso de acordes" o si quieres crear un programa el cual automaticamente escriba mas musica como es en mi caso.
</div>

## Musica Utilizada
<div style="text-align: justify">
El proyecto original utiliza música de Final Fantasy para entrenar su red. No he tendio la oportunidad de jugar muchos juegos de Final Fantasy solamente jugue el tercero. Asi que decidi utilizar musica de Final Fantasy 3, Chrono Trigger, Mario bros 3 y The Legend of Zelda: Ocarina of Time. La musica fue hallada en una pagina llamada <a href="https://www.vgmusic.com/">VGMusic</a> la cual contiene una variedad de musica de videojuegos en formato midi y tiene su propia seccion de midis que solo utilizan el piano lo cual fue muy util. Lamentablemente al buscar musica de Star Wars en formato midi no halle nada que fuera solamente el piano lo que hacia un poco mas dificil la idea de utilizarlo como datos de entrenamiento.
</div>

## El Modelo
<div style="text-align: justify">
El modelo de la red es muy similar al modelo original, los únicos cambios que hice fue aumentar el número de neuronas de las capas de LSTM y aumentar el dropout entre las capas
</div>

<p align="center">
  <img src="https://i.gyazo.com/8559083d1ead5a2499f8341bf6b63c76.png">
</p>

<div style="text-align: justify">
Además en cuanto al entrenamiento, intente un cambio con el numero batches pero con 32 me causaba resultados que no me gustaron asi que regrese a 64 y el número de epochs a 250.<br> Aparte del cambio al modelo, no hice muchos cambios que valgan la pena mencionar solamente cosas que ayudaron al entrenamiento
</div>

## Resultados
<div style="text-align: justify">
En mis primeras corridas solamente utilizaba musica de cada juego por separado, para ver que resultados me daban y al final utilize toda la musica que obtuve en una sola corrida para ver los resultados
</div>

### Chrono Trigger
<div style="text-align: justify">
Aqui tuve algunos problemas con las primeras dos corridas porque fue cuando estaba experimentando con el tamaño del batch en el ultimo resultado volvi a batch de 64 y siento que los resultados fueron mejores si han jugado Chrono Trigger podran ver las similitudes con las piezas originales, aqui la menor perdida fue de 0.027 con el epoch 243
</div><br>

Chrono Trigger con 100 epoch
<audio controls="">
<source src="assets/ChronoTrigger1.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

Chrono Trigger con 150 epoch
<audio controls="">
<source src="assets/ChronoTrigger2.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

Chrono Trigger con 250 epoch
<audio controls="">
<source src="assets/ChronoTrigger3.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

### Mario Bros
<div style="text-align: justify">
Los resultados fueron mejor de lo que esperaba que serian al iniciar esta parte del entrenamiento mi cancion favorita de las que genero fue la segunda a los que han jugado este juego veran muchas partes similares a las originales, aqui la menor perdida fue de 0.024 con el epoch 248
</div><br>

Mario Bros 3 con 100 epoch
<audio controls="">
<source src="assets/MarioBros1.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

Mario Bros 3 con 150 epoch
<audio controls="">
<source src="assets/MarioBros2.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

Mario Bros 3 con 250 epoch
<audio controls="">
<source src="assets/MarioBros3.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

### Legend of Zelda: Ocarina of Time
<div style="text-align: justify">
En este punto esperaba mucho ya que la musica de este juego es muy buena en mi opinion y no me defraudo el programa las musica quedo bien y en particular la segunda es muy similar a una cancion del juego, aqui la menor perdida fue de 0.029 con el epoch 248
</div><br>

The Legend of Zelda con 100 epoch
<audio controls="">
<source src="assets/Ocarina1.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

The Legend of Zelda con 150 epoch
<audio controls="">
<source src="assets/Ocarina2.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

The Legend of Zelda con 250 epoch
<audio controls="">
<source src="assets/Ocarina3.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

### Combinacion de toda la musica
<div style="text-align: justify">
No sabia que esperar de estos resultados, pensaba que no quedarian tan bien las canciones y que quiza haria una cancion bastante mala lo dejo a criterio de todos como quedo, tuvo una perdida de 0.8 en el epoch 250
</div><br>

<audio controls="">
<source src="assets/Combine.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

## Agradecimientos
- A Sigurður Skúli por su <a href="https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5">blog</a> sobre generacion de musica con LSTM
- A Christopher Olah por su <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">post</a> sobre redes recurrentes y LSTM me ayudo a entender bastante
- A el maestro Julio Waissman por sus clases sobre redes neuronales y apoyo con dudas
- A la maestra Sonia Sosa por su apoyo en como obtener la informacion y manejar mi metodo de trabajo
