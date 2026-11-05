# Aprendizado Conexionista: EXPLICAÇÃO DA REDE NEURAL DOS SLIDES DE FEI-FEI LEE**

Redes Neurais Conexionistas: Implementação de MLP em NumPy

Este documento descreve a implementação de uma Rede Neural Perceptron Multicamadas (MLP) utilizando apenas a biblioteca NumPy. O código demonstra como realizar a propagação direta (Forward Pass), o cálculo da perda (Loss) e o algoritmo de retropropagação (Backpropagation) de forma analítica, sem o uso de frameworks com diferenciação automática (como PyTorch ou TensorFlow).

# 1. Arquitetura e Dimensões

A rede possui uma arquitetura Feedforward Fully Connected (totalmente conectada) com 3 camadas:

* Tamanho do Lote (N): 64 amostras processadas em paralelo.
* Camada de Entrada (D_in): 1000 neurônios.
* Camada Oculta (H): 100 neurônios.
* Camada de Saída (D_out): 10 neurônios.

Inicialização dos Parâmetros:

* Entradas (x): Matriz de dimensão (64, 1000) gerada aleatoriamente.
* Alvos (y): Matriz de dimensão (64, 10) gerada aleatoriamente.
* Pesos da Camada 1 (w1): Matriz de dimensão (1000, 100).
* Pesos da Camada 2 (w2): Matriz de dimensão (100, 10).

# 2. Código-Fonte em Python

import numpy as np
from numpy.random import randn

**Definição das dimensões da rede:**
N, D_in, H, D_out = 64, 1000, 100, 10

**Inicialização aleatória dos dados e pesos:**
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

**Loop de treinamento (2000 épocas):**
```
for t in range(2000):
# 1. Forward Pass
h = 1 / (1 + np.exp(-x.dot(w1)))  # Ativação Sigmóide na camada oculta
y_pred = h.dot(w2)                 # Saída linear

# 2. Cálculo da Função de Perda (SSE - Erro Quadrático Total)
loss = np.square(y_pred - y).sum()
print(t, loss)

# 3. Backpropagation (Gradientes Analíticos)
grad_y_pred = 2.0 * (y_pred - y)
grad_w2 = h.T.dot(grad_y_pred)
grad_h = grad_y_pred.dot(w2.T)
grad_w1 = x.T.dot(grad_h * h * (1 - h))  # Derivada da Sigmóide: h * (1 - h)

# 4. Atualização dos Pesos (SGD)
w1 -= 1e-4 * grad_w1
w2 -= 1e-4 * grad_w2

```

# 3. Passo a Passo do Algoritmo

**Passo 1: Forward Pass (Propagação Direta)**

1. Camada Oculta (h): Aplica a combinação linear dos dados de entrada x com os pesos w1, seguida pela função de ativação Sigmóide:
h = 1 / (1 + exp(-x * w1))
2. Camada de Saída (y_pred): Combinação linear dos valores ativados h com os pesos w2 (ativação linear para regressão):
y_pred = h * w2

**Passo 2: Cálculo da Perda (Loss)**
Mede a divergência entre a previsão da rede (y_pred) e os rótulos reais (y) utilizando o Erro Quadrático Total (Sum of Squared Errors):
Loss = sum((y_pred - y)^2)

**Passo 3: Backpropagation (Retropropagação)**
Aplica a Regra da Cadeia para calcular a derivada parcial da função de perda em relação a cada conjunto de pesos:

* Gradiente da Saída: grad_y_pred = 2 * (y_pred - y)
* Gradiente em relação a w2: grad_w2 = h^T * grad_y_pred
* Gradiente em relação à Camada Oculta (h): grad_h = grad_y_pred * w2^T
* Gradiente em relação a w1: grad_w1 = x^T * (grad_h * h * (1 - h))
(Nota: O termo h * (1 - h) representa a derivada exata da função de ativação Sigmóide).

**Passo 4: Atualização dos Parâmetros**
Ajusta os pesos no sentido oposto ao gradiente usando Gradiente Descendente (SGD) com taxa de aprendizado de 0.0001 (1e-4):
w1 = w1 - 0.0001 * grad_w1
w2 = w2 - 0.0001 * grad_w2

# 4. Contexto Pedagógico

Este exemplo foi popularizado nos cursos de Visão Computacional e Deep Learning de Stanford (CS231n), liderados por Fei-Fei Lee e Justin Johnson. O objetivo principal do slide é demonstrar o cálculo explícito e manual de gradientes antes de apresentar aos estudantes as abstrações modernas de autodiferenciação fornecidas por bibliotecas como PyTorch (torch.autograd) e TensorFlow.
