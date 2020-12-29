#!/usr/bin/env python
# coding: utf-8

# ## Dados com muitas dimensões

# In[60]:


import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import random 
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

SEED = 20
random.seed(SEED)


uri = "https://raw.githubusercontent.com/DsGrilo/Alura_Care/main/data-set/exames.csv"

resultados_exames = pd.read_csv(uri)
resultados_exames.head()


# In[61]:


## Retorna a soma dos campos que estão como NaN -
resultados_exames.isnull().sum()


# In[ ]:





# In[62]:


## adiciona na variavel somente valores das colunas de exames, fazendo o drop das duas colunas 
valores_exames= resultados_exames.drop(columns = ['id', 'diagnostico']) 
valores_exames_v1 = valores_exames.drop(columns="exame_33")
## escolhe somente a coluna diagnostico do data
diagnostico= resultados_exames.diagnostico

## faz a segregação dos dados que serão usados para treino e para testes
treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v1, 
                                                        diagnostico,
                                                        test_size = 0.3)


# In[82]:


## n_estimator define quantas arvores de decisão serão criadas
classifier = RandomForestClassifier(n_estimators = 100,random_state= SEED )
## faz com que o modelo se adeque aos dados usados
classifier.fit(treino_x, treino_y)

print("Resultado da Classificação é %.2f%%" % (classifier.score(teste_x, teste_y) * 100))


# In[86]:


## Cria um "classificador burro" para ter como BASE LINE dos testes
classificador_bobo = DummyClassifier(strategy= "most_frequent")
classificador_bobo.fit(treino_x, treino_y)

print("Resultado da Classificação Bobo é %.2f%%" % (classificador_bobo.score(teste_x, teste_y) * 100))


# ## Avançando e Explorando os Dados

# In[ ]:




