#                                                   DISCUSSÃO DO exemplo 01

Esse código implementa e treina uma rede neural artificial usando TensorFlow para aprender a função cosseno. Ele serve como um exemplo didático de como redes neurais podem ser usadas para aproximação de funções (regressão).

Estou gerando 1000 valores de ângulo aleatórios entre 0 e 6π e calcula o cosseno desses valores, adicionando um pequeno ruído para simular dados reais.
O modelo tem uma rede neural com três camadas ocultas de 50 neurônios cada, usando a função de ativação tanh. A camada de saída tem um neurônio (regressão). Gera 100 valores de teste igualmente espaçados, calcula o cosseno verdadeiro e faz a previsão com a rede treinada.


Plota os dados de treinamento, a curva do cosseno verdadeiro e a curva prevista pela rede, permitindo comparar visualmente o desempenho do modelo.

O modelo é compilado com o otimizador Adam e função de perda MSE (erro quadrático médio). O treinamento ocorre por 1000 épocas.


#                                                   DISCUSSÃO DO exemplo 02

Esse código segue a mesma lógica e estrutura do exemplo 01, apenas trocando a função alvo de cosseno para seno.

Contudo, este código implementa e treina uma rede neural artificial usando TensorFlow para aprender a função seno. Ele gera dados de treinamento (ângulos e seus senos, com ruído), define e treina uma rede neural, faz previsões para novos ângulos, plota os resultados e calcula o erro quadrático médio.

#                                                   DISCUSSÃO DO exemplo 03


Para o exemplo 03, eu entendi que a taxa de aprendizado inicial (learning_rate_init) controla o tamanho dos passos que o modelo dá para ajustar os pesos. Quando essa taxa é aumentada, a convergência pode ser mais rápida, porém com maior risco de instabilidade. Já ao diminuí-la, o treinamento se torna mais lento, mas tende a ser mais estável.
O número máximo de iterações (max_iter), que define o máximo de interações o programa pode ter para chegar no seu objetivo. Aumentar esse número faz com que demore a refina sua resposta, enquanto reduzi-lo pode fazer o modelo parar antes de encontrar uma boa solução. Gostei de usar em conjunto os parâmetros n_iter_no_change e tol, que controlam quando o treinamento deve ser interrompido.
O hidden_layer_sizes, determina a arquitetura da rede e influencia a capacidade do modelo. Colocar mais camadas ou neurônios aumenta a complexidade e o poder de representação da rede pelo que entendi, pois nesse código não percebi uma diferença muito grande aumento ou diminuição no número de camadas permitindo que ela modele funções um pouco mais complicadas. Contudo, essa rede neural é pequena e o treinado é mais rápido e parece ser mais estável.
Eu não consegui um perfeito treinamento dessa rede neural e essa é a versão que melhor consegui deixar o código. (exemplo03_sklearn.ipynb). 

#                                                   DISCUSSÃO DO exemplo 04

Este código, é semenlhante aos anteriores, utiliza uma rede neural artificial com TensorFlow para aprender a função tangente. Ele gera dados de treinamento (ângulos e seus valores de tan(x), filtrando valores extremos para evitar problemas com as assíntotas), adiciona ruído, treina a rede neural, faz previsões para novos ângulos e compara visualmente os resultados previstos com os valores reais. Por fim, calcula o erro quadrático médio para avaliar o desempenho do modelo. É um exemplo prático de como redes neurais podem aproximar funções matemáticas mesmo em casos mais desafiadores, como a tangente.
Deixei dois exemplos nas imagens de como as epchos e as batch-size podem interferir nos resultados. 