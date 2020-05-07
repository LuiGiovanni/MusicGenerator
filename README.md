# Generando Musica con LSTM

## Una breve descripcion 
<div style="text-align: justify"> 
Buscando proyectos sobre aplicaciones de redes neuronales recurrentes, los principales temas que me llamaban la atencion eran sobre generación de música. Decidí centrarme en la generación de música y buscar proyectos relacionados a esto. En mi busqueda me encontré con esta entrada de blog en la que genera música de piano usando una red neuronal LSTM con Keras y entrenada con música de final fantasy, aunque estaba mas interesado en musica de Star Wars era un buen inicio.<br> Ahora, una intruducción hyper superficial sobre lo que son las redes neuronales recurrentes y LSTMs:
</div>

## Redes Neuronales Recurrentes
<div style="text-align: justify"> 
Una red neuronal recurrente es una red que tiene memoria interna. Gracias a esta memoria les es posible ser más precisa en predicciones sobre que es lo siguiente que debería suceder. Ya que estas redes pueden entender contexto de la información, son el tipo de red ideal cuando se está tratando con datos secuenciales: lenguaje hablado y escrito, música, películas, el clima, etc., lo cual tiene sentido cuando piensas que por ejemplo cuando hablas lo siguiente que vas a decir depende de el flujo de la conversación hasta el momento, o que el clima de mañana depende en parte del clima de hoy.<br> Los humanos no comienzan a pensar desde cero cada segundo. A medida que lees este blog, comprendes cada palabra en base a tu conocimiento de las palabras previas. No tiras todo y comienzas a pensar desde cero de nuevo. Tus pensamientos tienen persistencia.
</div>

<p align="center">
  <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png">
</p>

<div style="text-align: justify"> 
Algo curioso de estas redes neuronales es que el concepto no es nuevo. Desde 1980 parecen existir pero hasta hace pocos años se  ha tenido el suficiente poder de computo para hacerlas andar y que den resultados.
</div>

## LSTM (Long Short Term Memory)
<div style="text-align: justify">
Las redes LSTM son una variante de red neuronal recurrente. Estas redes surgieron principalmente para solventar un problema de dependencia de las redes recurrentes. Cuando el contexto necesario para predecir es de relativamente corto plazo, las redes recurrentes no tienen problemas, pero cuando se necesita de información que se remonte a entradas mucho más anteriores, se les dificulta predecir correctamente. No es que sea imposible, es técnicamente cierto que bajo ciertos pesos es posible que la red no tenga problemas de ese tipo, pero en la práctica esto no sucede. Aquí es donde entran las redes LSTM a salvar el día ya que fueron diseñadas para combatir justo ese problema.<br> Así como las redes recurrentes, las redes LSTM no son especialmenten nuevas. Existen desde más o menos 1990 pero han tenido que esperar hasta el poder de computo actual para poder mostrar buenos resultados.
</div>

<p align="center">
  <img src="https://i.pinimg.com/originals/27/95/bc/2795bc16b012322f7767cd4d940ba2e3.png">
</p>
