#!/usr/bin/env python
# coding: utf-8

# ## Dados com muitas dimensões

# In[69]:


import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import random 
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
import matplotlib.pyplot as plt
plt.figure('figure', figsize=(10, 10))

SEED = 20
random.seed(SEED)


uri = "https://raw.githubusercontent.com/DsGrilo/Alura_Care/main/data-set/exames.csv"

resultados_exames = pd.read_csv(uri)
resultados_exames.head()


# In[70]:


## Retorna a soma dos campos que estão como NaN -
resultados_exames.isnull().sum()


# In[71]:


## adiciona na variavel somente valores das colunas de exames, fazendo o drop das duas colunas 
valores_exames= resultados_exames.drop(columns = ['id', 'diagnostico']) 
valores_exames_v1 = valores_exames.drop(columns="exame_33")
## escolhe somente a coluna diagnostico do data
diagnostico= resultados_exames.diagnostico

## faz a segregação dos dados que serão usados para treino e para testes
treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v1, 
                                                        diagnostico,
                                                        test_size = 0.3)


# In[72]:


## n_estimator define quantas arvores de decisão serão criadas
classifier = RandomForestClassifier(n_estimators = 100,random_state= SEED )
## faz com que o modelo se adeque aos dados usados
classifier.fit(treino_x, treino_y)

print("Resultado da Classificação é %.2f%%" % (classifier.score(teste_x, teste_y) * 100))


# In[73]:


## Cria um "classificador burro" para ter como BASE LINE dos testes
classificador_bobo = DummyClassifier(strategy= "most_frequent")
classificador_bobo.fit(treino_x, treino_y)

print("Resultado da Classificação Bobo é %.2f%%" % (classificador_bobo.score(teste_x, teste_y) * 100))


# ## Avançando e Explorando os Dados

# In[74]:


## Chama o modelo de treino
padronizador = StandardScaler()
## Treina o modelo para seus dados
padronizador.fit(valores_exames_v1)
## atribui  a transformação para uma nova variavel
valores_exames_v2 = padronizador.transform(valores_exames_v1)

valores_exames_v2  = pd.DataFrame(data= valores_exames_v2, 
                                  columns= valores_exames_v1.keys())


# In[75]:


## Esta concatenando a coluna diagnostico junto aos valores_exames_v1
dados_plot = pd.concat([diagnostico, valores_exames_v2.iloc[:,0:10]], axis= 1 )
dados_plot.head()


# In[76]:


dados_plot = pd.melt(dados_plot, id_vars="diagnostico", var_name="exames", value_name="valores")
dados_plot.head()


# In[77]:


## Grafico estilo violina x= colunas exames y= valores e hue = diagnostico 
sns.violinplot(x = "exames", y = "valores",
               hue = "diagnostico",
               data = dados_plot,
               split= True)

## rotaciona os nomes do eixo X para a Vertical
plt.xticks(rotation = 90)


# In[81]:


def grafico_violino(values, inicio, fim):
    dados_plot = pd.concat([diagnostico, values.iloc[:,inicio:fim]], axis= 1 )
    dados_plot = pd.melt(dados_plot, id_vars="diagnostico", var_name="exames", value_name="valores")
    plt.figure(figsize=(10, 10))
    sns.violinplot(x = "exames", y = "valores",
                   hue = "diagnostico",
                   data = dados_plot,
                   split= True)
    plt.xticks(rotation = 90)

grafico_violino(valores_exames_v2, 10, 21)


# In[83]:


grafico_violino(valores_exames_v2, 21, 32)


# In[98]:


valores_exames_v3 = valores_exames_v2.drop(columns=["exame_29", "exame_4"])

def classificar(values):
    SEED = 20
    random.seed(SEED)
    treino_x, teste_x, treino_y, teste_y = train_test_split(values, 
                                                            diagnostico,
                                                            test_size = 0.3)
    classifier = RandomForestClassifier(n_estimators = 100,random_state= SEED )
    classifier.fit(treino_x, treino_y)

    print("Resultado da Classificação é %.2f%%" % (classifier.score(teste_x, teste_y) * 100))


# In[100]:


classificar(valores_exames_v3)


# ## Dados Correlacionados

# In[108]:


## traz uma matriz de correlação de valores
matriz_correlacao = valores_exames_v3.corr()

plt.figure(figsize= (17, 15))
## cria uma mapa de calor com a matriz de correlação
sns.heatmap(matriz_correlação, 
            annot= True,
            fmt= ".1f")


# In[110]:


## Guardas as variaveis com correlação acima de 0.99
matriz_correlacao_v1 = matriz_correlacao[matriz_correlacao > 0.99]

matriz_correlacao_v1 


# In[134]:


matriz_correlacao_v2 = matriz_correlacao_v1.sum()


# In[118]:


variaveis_correlacionadas = matriz_correlacao_v2[matriz_correlacao_v2 > 1]
variaveis_correlacionadas


# In[127]:


valores_exames_v4 = valores_exames_v3.drop(columns=variaveis_correlacionadas.keys())
valores_exames_v4


# In[129]:


classificar(valores_exames_v4)


# In[130]:


valores_exames_v5 = valores_exames_v3.drop(columns=["exame_3", "exame_24"])


# In[132]:


classificar(valores_exames_v5)


# In[ ]:




