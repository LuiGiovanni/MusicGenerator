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
  <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png" width="130" height="170">
</p>

<div style="text-align: justify"> 
Como lo muestra el diagrama de arriba un grupo de datos "X_t" entran a una parte de la red neuronal A y surge un resultado o como es llamado normalmente: "output" a la que indicamos como H_t el ciclo permite que esta informacion sea mandada de una parte de la red a otra, otra manera de ver esto suceder es asi:
</div>

<p align="center">
  <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png">
</p>

<div style="text-align: justify"> 
Esto es basicamente la primera imagen de arriba pero "desenrollada", es como una cadena de redes, se ve mas intuitivo no? Y la verdad las redes recurrentes son algo muy sencillo de entender, son algo muy intimamente relacionado con listas y sequencias. Lo mas sorprendente es que esta idea es vieja, la primera persona que llego a mencionar estas ideas data del año 1986 por Michael I. Jordan en una coferencia de ciencia cognitiva, el problema era que en esos tiempos las maquinas no daban para tanto poder de programacion, aun en estas fechas para proyectos muy grandes a veces necesitas una computadora mas poderosa. Pero aun asi el deseo de realizar un proyecto interesante gana y se han creado cosas bastante increibles con estas redes: reconocimiento de voces, modelado de lenguaje, traductores, etc. De hecho hay un articulo que habla sobre los resultados de gente que ha <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">trabajado con estas redes</a> se las recomiendo.
</div>

### Definiendo el problema a largo plazo de redes recurrentes

<div style="text-align: justify"> 
Una de las cosas mas llamativas en cuanto a redes recurrentes es cuando se quiere hacer una conexion entre informacion pasada e informacion presente, por ejemplo querer predecir la siguiente palabra dado una serie de palabras en un texto como "Mi camisa es de color azul" no es necesario mucho contexto, el problema se presenta cuando quieres hacer algo como "Mi familia es de Mexico...por eso hablo español" el objetivo en este caso es querer indicar que el idoma hablado es español pero si querermos reducir las posibilidades y hallar la palabra "español" debemos tener mas contexto, especificamente es de Mexico. Desafortunadamente entre mas crezca la informacion, mas dificil sera para la red aprender a conectar la informacion.<br><br> La teoria establece que una red recurrente podria resolver estos tipos de problemas pero solamente si el programador escoge perfectamente los parametros, afortunadamente las LSTM no tienes estas limitaciones!
</div>

## LSTM (Long Short Term Memory)
<div style="text-align: justify">
Estas redes fueron creadas explicitamente para resolver los problemas ya descritos de las rede recurrentes, pero como es que funcionan? Y cuales son la diferencias entre las dos? Pues, una red recurrente y una LSTM amba tienen el diseño de cadena antes mostrado, la red recurrente normalmente tiene una estructura bastante simple como una simple capa tanh
</div>

<p align="center">
  <img src="https://i.pinimg.com/originals/27/95/bc/2795bc16b012322f7767cd4d940ba2e3.png">
</p>

<div style="text-align: justify">
Aqui viene lo bueno, una red LSTM tiene una estructura mas compleja, dentro de cada modulo tiene cuatro capas que interactuan de una manera muy especial! Lo principal en las LSTM es un estado de cada celda o "cell state" (C_t) esta es la linea horizontal que corre en la parte superior pienselo como una cinta transportadora la cual lleva los cell states al final con cambios minimos, la LSTM puede hacerle cambios a los cell states, regulado por algo llamado "gates" o compuertas las cuales se usan para opcionalmente dejar pasar informacion. Estan compuestas por una capa de red neuronal sigma y un operacion producto-punto. El sigma envia un valor entre 0 y 1 el cual indica que tanto de cada componente dejar pasar con 0 siendo "no dejes pasar nada", los LSTM tienen tres de estas computas para proteger muy bien la informacion del cell state.
</div>


### LSTM paso a paso
<div style="text-align: justify">

</div>

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
