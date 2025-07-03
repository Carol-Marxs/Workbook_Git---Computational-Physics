#                                                    Discussão do passo 01, 02 e 03

Este código é um pouco mais "simples", ele simula o resfriamento de uma caneca de café usando a lei de resfriamento de Newton. Ele resolve a equação diferencial tanto de forma analítica quanto numérica (método RK4), gera dados sintéticos com ruído para simular medições reais e plota os resultados para comparar a solução teórica com os dados simulados. O objetivo é ilustrar como modelar, resolver e visualizar processos físicos.    




#                                                    Discussão do passo 4

Na atividade 4, tive dificuldades para implementar esse código, por isso optei por usar o TensorFlow. Meu ajuste de dados apresentava inconsistências, o que impedia o treinamento adequado da rede neural. Iniciei o processo importando as bibliotecas necessárias e definindo as condições do problema. Em seguida, calculei a solução analítica do problema inicial e gerei dados sintéticos com ruídos no intervalo de 0 a 300 segundos. Esses ruídos simulam medições experimentais reais, ajudando a rede a aprender o comportamento físico do sistema de forma mais realista.

Para estabilizar o treinamento e evitar que valores extremos de tempo prejudicassem a convergência do modelo, realizei a normalização da entrada temporal. Depois, construí a rede neural com uma camada de entrada e duas camadas ocultas, cada uma com 80 neurônios (sei que o professor pode me chamar a atenção pela quantidade de neurônios, mas foi o melhor ajuste que consegui alcançar, então deixei esse valor).

Implementei uma função de perda personalizada para forçar a rede a respeitar as leis físicas, combinando o erro nos dados com o erro da equação diferencial, garantindo assim uma solução fisicamente coerente. Por fim, compilei e treinei o modelo. Para avaliar a capacidade da rede de prever o comportamento além do intervalo de treino e testar sua capacidade de generalização e extrapolação física, realizei a predição para o intervalo de tempo de 0 a 1000 segundos.

#                                                    Discussão do passo 5

Na atividade 5, a principal mudança em relação à atividade anterior foi ampliar o intervalo de dados sintéticos de 0 até 1000 segundos. Isso significa que, agora, a rede neural tem acesso a uma janela de tempo maior durante o treinamento, o que influencia diretamente sua capacidade de aprendizado e de generalização.

Assim como na atividade anterior, optei por usar o TensorFlow devido à familiaridade e à flexibilidade da biblioteca. Iniciei o processo importando as bibliotecas necessárias e definindo as condições do problema. Em seguida, calculei a solução analítica da equação de resfriamento e gerei os dados sintéticos com ruídos no intervalo estendido (0 a 1000 s). Esse ruído foi adicionado para tornar os dados mais realistas, simulando imprecisões experimentais.

Para garantir a estabilidade do treinamento da rede, normalizei os valores de tempo. A rede neural foi construída com uma camada de entrada e duas camadas ocultas de 80 neurônios cada — um valor que, apesar de elevado, se mostrou eficaz para capturar o comportamento da solução no intervalo considerado.

Implementei uma função de perda personalizada que combina o erro dos dados com o erro da equação diferencial, garantindo que a solução aprendida respeite a física do problema. Após compilar e treinar o modelo, fiz a predição para o tempo de 0 a 1000 segundos — o mesmo intervalo usado para treinamento —, com o objetivo de verificar se a rede foi capaz de aprender corretamente todo o comportamento do sistema físico nesse intervalo ampliado.

#                                                    Discussão do passo 6

Aqui foi implementado a PINN sem conhecer o valor da taxa r, onde a rede foi capaz de descobrir o valor correto. 

A PINN se destacou por incorporar o conhecimento da equação física ao treinamento, permitindo estimar corretamente o parâmetro de resfriamento mesmo sem conhecê-lo previamente. Comparada à solução analítica, a PINN apresentou excelente aderência ao comportamento teórico. Em relação à NN de regressão simples, a PINN mostrou melhor generalização fora do intervalo de treino, enquanto a regressão simples se ajusta bem apenas aos dados disponíveis, sem garantir respeito à física do problema.
