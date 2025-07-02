#                                       DISCUSSÕES DO EXERCÍCIO 1 

Nas configurações iniciais do exercício 1, o gráfico apresenta um caminho até chegar no mínomo da função rente a forma de linha da gráfico, ponto a ponto. Minha primeiro modificação foi fixar epsilon = 0.01 e variar o alpha em 0.3 e 0.5. 
Em alpha = 0.3, a distância dos pontos é maior até chegar no mínimo da função, diminuindo assim, a quantidade de pontos, quando comparado ao alpha = 0.1.
Em alpha = 0.5, a distândia dos pontos ainda maior, o mínimo da função é encontrado, mas há apenas dois pontos, o incial de x0 = 5 e xf = 0.
Concluindo assim, que alpha controla o tamanho do passo que o ponto dar na direção oposta ao gradiente da função que está sendo minimizada.
Quando alpha é pequeno a convergência é mais lenta, mas mais segura e precisa. Alpha grande os passos grandes pode convergir mais rápido, mas também pode saltar o mínimo, podendo assim, perder informação.

Agora fixando alpha = 0.1 e variando epsilon em 0.1 e 0.001.
Epsilon = 0.1 é longe de zero, quando comparado ao 0.001 ou 0.01. 
sCom isso, epsilon define quando o algoritmo deve parar, quando consideramos que estamos suficientemente próximos do mínimo.


#                                       DISCUSSÕES DO EXERCÍCIO 2

 # $$O que acontece? O que você pode concluir sobre a escolha da taxa de aprendizado α?

 Como diz no anuciado do exercício, o gráfico dessa função exibe dois minímos globais, plotei simutaneamente os dois mínimos onde a escolha de alpha muda o passo para chegar nos mínimos. Os valores de alpha afeta fortemente a convergência, se muito pequeno a partícula converge lentamente, mas se grande, a partícula oscila até divergir. Mas há um meio termo onde a particula pode chegar rapidamente e com estabilidade. 


#                                        DISCUSSÕES DO EXERCÍCIO 3

Mantendo os mesmos parametros do experimento 2, mas para essa função, há uma adição de 1/4 deixando o lado direito do gráfico mais "alto".
Também mantendo a importância da escolha de alpha. Tive a liberdade de plotar as duas trajetorias com um alpha diferente para eles, ficou interessante a visualização.


#                                       DISCUSSÕES DO EXERCÍCIO 4

# EXERCÍCIO 4: O que acontece se você aumentar muito a taxa de aprendizado? E se você diminuir muito? Você consegue atingir o mínimo global?

Para um alpha muito grande, os passos são grandes e a partícula chega mais rápido até o mínimo, mas não é fiel a forma de linha do gráfico (perdendo informação). Para alpha muito pequeno a partícula demora para chegar no mínimo, mas está rende a forma de linha do gráfico, não perdendo quase que nenhuma informação. Sobre o mínimo global, considero alpha = 0.05 bem equilibrado, a partícula desce suave e relativamente rápido. Variando valores r0 pode ser possível sim encontrar o mínimo global. Minha sujestão (através) de tentativas r0 = [-1.5, 0.1], o máximo que consegui chegar. 
 