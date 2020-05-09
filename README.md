# Generando Musica con LSTM

## Una breve introduccion 
<div style="text-align: justify">  
A lo largo de los años hemos estado utilizado redes neuronales para mejorar muchos aspectos de nuestras vidas: desde recomendaciones de artículos que deseamos comprar, generar texto al estilo de algún autor en particular y hasta podemos <a href="https://jjups96.github.io/fast-style-transfer/">cambiar el estilo del arte de alguna imagen</a>. Mas recientemente, ha habido bastantes tutoriales y cursos de como generar texto a partir de redes neuronales y bastantes blogs de personas generando cambios de imágenes pero no mucho sobre generar música con redes neuronales, ahora esto no será un tutorial como tal, más que nada mi enfoque es mostrarles este mundo tan interesante e invitarlos a que se sumen a este grupo de personas que estudian o intentan mejorar el mundo de las redes neuronales.<br>
</div>

## Porque generar musica?
<div style="text-align: justify">   
El principal tema que me llamaba la atención era sobre generación de música, tenía tiempo con el deseo de realizar un proyecto similar más que nada porque los modelos que pueden utilizarse son muy interesantes y divertidos de poner en práctica. Busque unos proyectos relacionados a esto. A los inicios del proyecto mi idea era generar música de Star Wars así que empecé mi búsqueda por el sagrado y vasto mundo de Google como todo buen programador, encontré unos buenos e interesantes proyectos: uno utilizaba un modelo muy complejo que se basaba en un artículo publicado por un estudiante el modelo se llamaba C-RNN-GAN (Continuous Recurrent Neural Network Generative Adversial Model) esta sopa de letras pueden verla <a href="https://github.com/olofmogren/c-rnn-gan">aqui</a> este blog y repositorio fueron hechos por <a href="https://github.com/olofmogren">olofmorgen</a> les recomiendo checar su GitHub y ver sus proyectos son muy interesantes, pero al intentar estudiar el modelo y entenderle y con apoyo de mis maestros me di cuenta que este modelo aunque interesante era muy complicado para el tiempo que tenía así que decidí buscar otro y al final me encontré con un modelo llamado LSTM en un post el cual generaba música de videojuegos lo leí y me pareció bastante interesante.
</div>

## Un poco de explicacion.
<div style="text-align: justify">
Ahora antes de comenzar debemos entender un poco de terminología y entender algunos conceptos sobre redes neuronales que serán necesarios entender por lo menos a un nivel superficial.
</div>
   
### Redes Neuronales Recurrentes
<div style="text-align: justify"> 
En resumen, son redes que contienen un ciclo y permiten que información persista, Los humanos no comienzan a pensar desde cero cada segundo. A medida que lees este blog, comprendes cada palabra en base a tu conocimiento de las palabras previas. No tiras todo y comienzas a pensar desde cero de nuevo. Tus pensamientos tienen persistencia.<br><br> redes tradicionales no tienen este beneficio lo cual podría considerarse un gran defecto, por ejemplo, digamos que ves una película y quieres clasificar cada punto de esta, una red neuronal no podría razonar puntos previos de la película para informar de los nuevos puntos.
</div>

<p align="center">
  <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png" width="130" height="170">
</p>

<div style="text-align: justify"> 
Como lo muestra el diagrama de arriba un grupo de datos "X_t" entran a una parte de la red neuronal A y surge un resultado o como es llamado normalmente: "output" a la que indicamos como H_t el ciclo permite que esta información sea mandada de una parte de la red a otra, otra manera de ver esto suceder es así:
</div>

<p align="center">
  <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png">
</p>

<div style="text-align: justify">
Esto es básicamente la primera imagen de arriba, pero desarrollado, es como una cadena de redes, ¿se ve más intuitivo no? Las redes recurrentes desde un punto de vista práctico son algo muy íntimamente relacionado con listas y secuencias. Lo más sorprendente es que esta idea es vieja, la primera persona que llego a mencionar estas ideas data del año 1986, el problema era que en esos tiempos las maquinas no daban para tanto poder de programación, aun en estas fechas para proyectos muy grandes a veces necesitas una computadora más poderosa. Pero aun así se han creado proyectos bastante increíbles con estas redes: reconocimiento de voces, modelado de lenguaje, traductores, etc. De hecho, hay un artículo que habla sobre los resultados de gente que ha <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">trabajado con estas redes</a> se las recomiendo.
</div>

#### Definiendo el problema a largo plazo de redes recurrentes

<div style="text-align: justify">   
Una de las cosas más llamativas en cuanto a redes recurrentes es cuando se quiere hacer una conexión entre información pasada e información presente, por ejemplo querer predecir la siguiente palabra dado una serie de palabras en un texto como "Mi camisa es de color azul" no es necesario mucho contexto, el problema se presenta cuando quieres hacer algo como "Mi familia es de México...por eso hablo español" el objetivo en este caso es querer indicar que el idioma hablado es español pero si queremos reducir las posibilidades y hallar la palabra "español" debemos tener más contexto, específicamente que es de México. Desafortunadamente entre más crezca la información, más difícil será para la red aprender a conectar la información.<br><br> La teoría establece que una red recurrente podría resolver estos tipos de problemas, pero solamente si el programador escoge perfectamente los parámetros, ¡afortunadamente las LSTM no tienes estas limitaciones!
</div>

### LSTM (Long Short Term Memory)
<div style="text-align: justify">
Estas redes fueron creadas explícitamente para resolver los problemas ya descritos de las redes recurrentes, pero ¿cómo es que funcionan? ¿Y cuáles son las diferencias entre las dos? Pues, una red recurrente y una LSTM ambas tienen el diseño de cadena antes mostrado, la red recurrente normalmente tiene una estructura bastante simple como una simple capa tanh
</div>

<p align="center">
  <img src="assets/LSTM3-SimpleRNN.png">
</p>

<div style="text-align: justify">
Aquí viene lo bueno, una red LSTM tiene una estructura más compleja, dentro de cada módulo tiene cuatro capas que interactúan de una manera muy especial, lo principal en las LSTM es un estado de cada celda o "cell state" (C_t) esta es la línea horizontal que corre en la parte superior piénselo como una cinta transportadora la cual lleva los cell states al final con cambios mínimos, la LSTM puede hacerle cambios a los cell states, regulado por algo llamado "gates" o compuertas las cuales se usan para opcionalmente dejar pasar información. Están compuestas por una capa de red neuronal sigma y una operación producto-punto. La sigma envía un valor entre 0 y 1 el cual indica que tanto de cada componente dejar pasar con 0 siendo "no dejes pasar nada", los LSTM tienen tres de estas compuertas para proteger muy bien la información del cell state.
</div>

<p align="center">
  <img src="assets/LSTM3-chain.png">
</p><br>

<div style="text-align: justify">  
Explicar paso a paso como funciona un LSTM toman mucho tiempo, para más información recomiendo que vean el blog de <a href="https://github.com/colah/">colah</a> donde explica todo más a fondo <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">aqui</a>
</div>

### Music21
<div style="text-align: justify">
Lo que hace el proyecto es leer la música en formato MIDI usando music21 para extraer las notas de los archivos MIDI con los que se quiere aprender para generar música similar.<br><br> Music21 es una colección de herramientas para ayudar a estudiantes y otros hallar respuestas de música más fácilmente, cosas como "Me pregunto cuántas veces Bach hace eso" o "desearía conocer cual banda fue la primera en usar ese progreso de acordes" o si quieres crear un programa el cual automáticamente escriba más música como es mi caso.
</div>

### Keras
<div style="text-align: justify">
Básicamente es una API de redes neuronales de alto nivel. Esta simplifica interacciones con Tensorflow, se creó con el enfoque de poder realizar experimentaciones rápidas. El autor original utilizo Keras para poder crear y entrenar el modelo LSTM, ya que el modelo es entrenado se utiliza para generar notación musical para nuestra música.
</div>

## Musica Utilizada
<div style="text-align: justify">
El proyecto original utiliza música de Final Fantasy para entrenar su red. No he tenido la oportunidad de jugar muchos juegos de Final Fantasy solamente he podio jugar el tercero. así que decidí utilizar música de Final Fantasy 3, Chrono Trigger, Mario bros 3 y The Legend of Zelda: Ocarina of Time. La música fue hallada en una página llamada <a href="https://www.vgmusic.com/">VGMusic</a> la cual contiene una variedad de música de videojuegos en formato midi y tiene su propia sección de midis que solo utilizan el piano lo cual fue muy útil. Lamentablemente al buscar música de Star Wars en formato midi no halle nada que fuera solamente el piano lo que hacía un poco más difícil la idea de utilizarlo como datos de entrenamiento.
</div>

## El Modelo
<div style="text-align: justify">
Primero les diré muy básico que es lo que está pasando con el código: empieza obteniendo la música y utilizando Music21 para convertir las notas de la música a datos que son más manejables por el programa, los guarda en un archivo llamado "data/notes" esto lo hace cada vez que corres el programa así que si cambias la música en tu carpeta "midi" volverá a convertir esa música en data para poder entrenarla, lo siguiente que hace es preparar las secuencias que serán usadas por la red neuronal obteniendo todos los "pitch names" de las notas que ya obtuvimos crea un diccionario y mapea los pitch a enteros, después de esto crea secuencias de inputs y sus outputs correspondientes después hace un reshape de numpy al input para algo más compatible con capas LSTM y normaliza el input.<br><br> Ahora lo interesante y donde estarán haciendo cambios para entender cómo funciona la red, vamos a crear la estructura de la red neuronal. y al final entrenamos nuestra red.
</div><br>

<p align="center">
  <img src="https://i.gyazo.com/8559083d1ead5a2499f8341bf6b63c76.png">
</p>

<div style="text-align: justify">
Cabe aclarar que mucho del proyecto no fue hecho desde cero por mí, me apoye mucho de trabajo de otras personas que han venido desarrollando este tipo de proyectos desde hace tiempo. El código original lo pueden hallar en la sección de agradecimiento en el blog y pagina de towards data science de <a href="https://towardsdatascience.com/@sigurdurssigurg">Sigurður Skúli</a>.<br><br> En cuanto a mi entrenamiento, hice unas pruebas con el pipeline cambiando batches, pero con 32 me causaba resultados que no me gustaron así que fui a 64 y el número de epochs a 250 pude haber aumentado el numero a algo más alto para ver los resultados, pero afectaba mucho el tiempo para entregar el proyecto, mi plan a futuro es hacer entrenamientos con epochs más largos para hallar un punto donde sienta que es muy razonable el cambio o que la perdida será mínima. Mucho de mi enfoque se centro en cambiar las capas: la cantidad de capas y la cantidad de neuronas por cada capa ya que mucho del entrenamiento y pipeline ya eran bastante buenos los cambios no fueron tan drásticos como esperaba que lo serian al iniciar.<br><br> Otro defecto (En mi opinión) que tiene ahorita el código es que cuando termina el entrenamiento te genera un numero de archivos hdf5 igual a el numero epoch, el título de este archivo tiene el número de epoch y la perdida y cada vez que corras el predict.py para generar una nueva canción debes cambiar el archivo que busca a el nuevo archivo hdf5. Esto a mí no me gustaba ya que es muy incómodo así que lo cambie para que el nombre del archivo no deba cambiarse.
</div>

## Resultados
<div style="text-align: justify">
En mis primeras corridas solamente utilizaba música de cada juego por separado, para ver qué resultados me daban y al final utiliza toda la música que obtuve en una sola corrida para ver los resultados
</div>

### Chrono Trigger
<div style="text-align: justify">
Aquí tuve algunos problemas con las primeras dos corridas porque fue cuando estaba experimentando con el tamaño del batch en el último resultado volví a batch de 64 y siento que los resultados fueron mejores si han jugado Chrono Trigger podrán ver las similitudes con las piezas originales, aquí la menor perdida fue de 0.027 con el epoch 243
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
Los resultados fueron mejor de lo que esperaba que fueran al iniciar esta parte del entrenamiento mi canción favorita de las que genero fue la segunda a los que han jugado este juego verán muchas partes similares a las originales, aquí la menor perdida fue de 0.024 con el epoch 248
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
En este punto esperaba mucho ya que la música de este juego es muy buena en mi opinión y no me defraudo el programa la música quedo bien y en particular la segunda es muy similar a una canción del juego, aquí la menor perdida fue de 0.029 con el epoch 248
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
No sabía que esperar de estos resultados, pensaba que no quedarían tan bien las canciones y que quizá haría una canción bastante mala lo dejo a criterio de todos como quedo, tuvo una pérdida de 0.8 en el epoch 250
</div><br>

<audio controls="">
<source src="assets/Combine.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio><br>

## Conclusion
<div style="text-align: justify">
Al final de todo me siento muy satisfecho con los resultados que me dio el proyecto y me siento satisfecho con el trabajo que he realizado el tema de las redes recurrentes y machine learning están en un puto muy temprano y emocionante, a todos los que lean este blog se los recomiendo y espero que encuentren un gusto como el que he encontrado yo. Todos los enlaces llevan a lecturas muy interesantes de personas que han hecho un esfuerzo y aunque no vayan a trabajar en esto les recomiendo estas lecturas y estudiar más del tema se darán cuenta que mucho de la vida cotidiana utiliza redes recurrentes y machine learning en general. Si quieren empezar probando el código pueden correr predict.py para que les genere algunas canciones y si quieren probar el entrenamiento pueden reducir el número de epochs a algo más razonable y moverles a los valores para ir entendiendo que está sucediendo con el código, se los recomiendo.
</div><br>

## Agradecimientos
- A Sigurður Skúli por su <a href="https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5">blog</a> sobre generacion de musica con LSTM, el codigo original tambien pueden hallarlo aqui
- A Christopher Olah por su <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">post</a> sobre redes recurrentes y LSTM me ayudo a entender bastante y por las imagenes que fueron creadas por el.
- A el maestro Julio Waissman por sus clases sobre redes neuronales y apoyo con dudas
- A la maestra Sonia Sosa por su apoyo en como obtener la informacion y manejar mi metodo de trabajo
