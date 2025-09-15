# Tentativa 1 - Funkpop Júlio

## Etapa 1 - Coleta das Imagens e Preparação
ambiente: LSC Unicamp de fundo, em cima de uma escrivaninha, Funkopop do Júlio Avelar  
dispositivo: Xiaomi Redmi Note 13 Pro 5g  
num fotos: 108  
EXIF: irrelevante  
Estratégia captura: Força bruta. Quanto mais imagens e mais repetição de certas partes da figura, menor a chance de faltar algo importante no futuro.
Desafios: Interpretação do enunciado. O que é um fundo com textura? O meu fundo tem textura? quanto é uma sobreposição de 60-80%? Minhas fotos têm isso? Quanto eu deveria variar de altura e angulação?  


## Etapas 2 e 3 - Detecção e Emparelhamento de características
Utilizamos SIFT e ORB - Comparação entre duas imagens (img7 e img12)
Features extraídas com SIFT: 954 e 1142 features
Features extraídas com ORB: 2925 e 3710 features
Matches obtidos com SIFT: 145 matches bons
Matches obtidos com ORB: 117 matches bons
Escolhemos SIFT por ter mais matches bons, ser mais fácil de utilizar por ter mais ferramentas que o suportam e por ter visualmente matches melhores nas fotos resultantes.
[Foto das features do SIFT e ORB, foto dos matches SIFT e ORB]

Para emparelhamento, utilizamos FLANN e o ratio test de Lowe para filtragem

Utilizamos RANSAC para F/E e recuperamos as poses da câmera na figura 1 com triangulação já feita.
[Foto das poses da câmera, quadrado com pointinhos.]

# Etapa 4 - Reconstrução 3D e Densificação

Executamos SfM via COLMAP
Executamos MVS
não geramos malha e texturização
# REALIZAR LIMPEZA DOS PONTOS SUJOS!


Desafios enfrentados:

Das 108 imagens que tiramos, o COLMAP descartou a maioria delas, provavelmente por baixa qualidade (Acreditamos que, se o COLMAP não identifica features/matches o suficiente em uma imagem, a mesma é desconsiderada), e ficamos apenas com 35 imagens para o resultado final, a grande maioria das mesmas tirando uma foto frontal do funkopop, o que fez com que as costas da figura ficassem incompletas, além de abaixar a qualidade da imagem como um todo.

Não conseguimos rodar com ORB por falta de suporte do COLMAP à esse extrator.

O uso do COLMAP tirou bastante nosso controle sobre vários aspectos do resultado final, o que, ao mesmo tempo que facilitou, também dificulta em conseguir um resultado melhor trabalhado.

A forma como as fotos foram tiradas causaram um impacto forte na qualidade das mesmas, além do ambiente em que foram feitas. Houve muito ruido na hora de visualizar a imagem no Open3D devido a objetos não relacionados no ambiente ao redor, como carregador de notebook, folha de papel, monitor, etc. Para próximas coletas, seria ideal um ambiente mais limpo, com acessibilidade 360 graus para facilitar tirar as fotos das costas e não ocasionar problemas de uma parte da imagem ser perdida devido à descarte de fotos de má qualidade.

Tirar melhor as fotos, basicamente.

# Tentativa 2 - Astronauta sem textura

## Etapa 1 - Coleta das Imagens e Preparação
ambiente: LSC Unicamp de fundo, em cima de uma mesa circular vazia, Astronauta sem textura do Júlio Avelar feito em impressora 3D  
dispositivo: Xiaomi Redmi Note 13 Pro 5g  
num fotos: 121  
EXIF: irrelevante  
Estratégia captura: Força bruta com considerações em relação às falhas da tentativa anterior. Figura colocada em mesa circular acessível por todos os lados para facilitar fotos de todos os pontos de vista, fotos tiradas em ordem e com bastante sobreposição para melhor organização, objetos irrelevantes para o panorama foram retirados das redondezas para não causarem ruídos.

Desafios: Angulação consistente. Mesmo com mais acessibilidade ao redor da figura, ainda é difícil manter a distância da câmera e a angulação consistentes entre fotos adjacentes. Além disso, foi escolhido, propositalmente, uma figura monocromática branca para avaliar o quanto de um impacto isso têm no resultado final.