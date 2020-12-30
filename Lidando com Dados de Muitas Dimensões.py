#!/usr/bin/env python
# coding: utf-8

# ## Dados com muitas dimensões

# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import random 
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt
plt.figure('figure', figsize=(10, 10))

SEED = 20
random.seed(SEED)


uri = "https://raw.githubusercontent.com/DsGrilo/Alura_Care/main/data-set/exames.csv"

resultados_exames = pd.read_csv(uri)
resultados_exames.head()


# In[2]:


## Retorna a soma dos campos que estão como NaN -
resultados_exames.isnull().sum()


# In[3]:


## adiciona na variavel somente valores das colunas de exames, fazendo o drop das duas colunas 
valores_exames= resultados_exames.drop(columns = ['id', 'diagnostico']) 
valores_exames_v1 = valores_exames.drop(columns="exame_33")
## escolhe somente a coluna diagnostico do data
diagnostico= resultados_exames.diagnostico

## faz a segregação dos dados que serão usados para treino e para testes
treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v1, 
                                                        diagnostico,
                                                        test_size = 0.3)


# In[4]:


## n_estimator define quantas arvores de decisão serão criadas
classifier = RandomForestClassifier(n_estimators = 100,random_state= SEED )
## faz com que o modelo se adeque aos dados usados
classifier.fit(treino_x, treino_y)

print("Resultado da Classificação é %.2f%%" % (classifier.score(teste_x, teste_y) * 100))


# In[5]:


## Cria um "classificador burro" para ter como BASE LINE dos testes
classificador_bobo = DummyClassifier(strategy= "most_frequent")
classificador_bobo.fit(treino_x, treino_y)

print("Resultado da Classificação Bobo é %.2f%%" % (classificador_bobo.score(teste_x, teste_y) * 100))


# ## Avançando e Explorando os Dados

# In[6]:


## Chama o modelo de treino
padronizador = StandardScaler()
## Treina o modelo para seus dados
padronizador.fit(valores_exames_v1)
## atribui  a transformação para uma nova variavel
valores_exames_v2 = padronizador.transform(valores_exames_v1)

valores_exames_v2  = pd.DataFrame(data= valores_exames_v2, 
                                  columns= valores_exames_v1.keys())


# In[7]:


## Esta concatenando a coluna diagnostico junto aos valores_exames_v1
dados_plot = pd.concat([diagnostico, valores_exames_v2.iloc[:,0:10]], axis= 1 )
dados_plot.head()


# In[8]:


dados_plot = pd.melt(dados_plot, id_vars="diagnostico", var_name="exames", value_name="valores")
dados_plot.head()


# In[9]:


## Grafico estilo violina x= colunas exames y= valores e hue = diagnostico 
sns.violinplot(x = "exames", y = "valores",
               hue = "diagnostico",
               data = dados_plot,
               split= True)

## rotaciona os nomes do eixo X para a Vertical
plt.xticks(rotation = 90)


# In[10]:


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


# In[11]:


grafico_violino(valores_exames_v2, 21, 32)


# In[12]:


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


# In[13]:


classificar(valores_exames_v3)


# ## Dados Correlacionados

# In[15]:


## traz uma matriz de correlação de valores
matriz_correlacao = valores_exames_v3.corr()

plt.figure(figsize= (17, 15))
## cria uma mapa de calor com a matriz de correlação
sns.heatmap(matriz_correlacao, 
            annot= True,
            fmt= ".1f")


# In[16]:


## Guardas as variaveis com correlação acima de 0.99
matriz_correlacao_v1 = matriz_correlacao[matriz_correlacao > 0.99]

matriz_correlacao_v1 


# In[17]:


matriz_correlacao_v2 = matriz_correlacao_v1.sum()


# In[18]:


variaveis_correlacionadas = matriz_correlacao_v2[matriz_correlacao_v2 > 1]
variaveis_correlacionadas


# In[19]:


valores_exames_v4 = valores_exames_v3.drop(columns=variaveis_correlacionadas.keys())
valores_exames_v4


# In[20]:


classificar(valores_exames_v4)


# In[21]:


valores_exames_v5 = valores_exames_v3.drop(columns=["exame_3", "exame_24"])


# In[22]:


classificar(valores_exames_v5)


# ## Automatizando a Seleção 

# In[26]:


selecionar_kmelhores = SelectKBest(chi2, k = 5)


# In[39]:


valores_exames_v6 = valores_exames_v1.drop(columns = ["exame_4", "exame_29", "exame_3", "exame_24"])


# In[41]:


treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size = 0.3)


# In[45]:


selecionar_kmelhores.fit(treino_x, treino_y)
treino_kbest = selecionar_kmelhores.transform(treino_x)
teste_kbest = selecionar_kmelhores.transform(teste_x)
treino_kbest.shape


# In[50]:


SEED = 20
random.seed(SEED)

classifier = RandomForestClassifier(n_estimators=100, random_state=1234)
classifier.fit(treino_kbest, treino_y)
print("Resultado da Classificação é %.2f%%" % (classifier.score(teste_kbest, teste_y) * 100))


# In[52]:


matriz_confusao = confusion_matrix(teste_y, classifier.predict(teste_kbest))


# In[54]:


matriz_confusao


# In[56]:


plt.figure(figsize = (10, 8))
sns.set(font_scale= 3)
sns.heatmap(matriz_confusao, 
            annot= True,
            fmt= "d").set(xlabel = "Predição", ylabel = "Real")


# # 0 - Cancer Benigno , 1 - Cancer Maligno

# ## Seleção com RFE

# In[61]:


from sklearn.feature_selection import RFE

SEED = 20
random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size = 0.3)

classifier = RandomForestClassifier(n_estimators=100, random_state=1234)
classifier.fit(treino_x, treino_y)

selecionar_rfe = RFE(estimator= classifier, n_features_to_select=5, step=1)
selecionar_rfe.fit(treino_x, treino_y)
treino_rfe = selecionar_rfe.transform(treino_x)
teste_rfe = selecionar_rfe.transform(teste_x)
classifier.fit(treino_rfe, treino_y)

matriz_confusao = confusion_matrix(teste_y, classifier.predict(teste_rfe))

plt.figure(figsize = (10, 8))
sns.set(font_scale= 3)
sns.heatmap(matriz_confusao, 
            annot= True,
            fmt= "d").set(xlabel = "Predição", ylabel = "Real")

print("Resultado da Classificação é %.2f%%" % (classifier.score(teste_rfe, teste_y) * 100))


# ## Seleção com RFECV

# In[63]:


from sklearn.feature_selection import RFECV

SEED = 20
random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size = 0.3)

classifier = RandomForestClassifier(n_estimators=100, random_state=1234)
classifier.fit(treino_x, treino_y)

selecionar_rfecv = RFECV(estimator= classifier, cv = 5, step=1, scoring="accuracy")
selecionar_rfecv.fit(treino_x, treino_y)
treino_rfecv = selecionar_rfecv.transform(treino_x)
teste_rfecv = selecionar_rfecv.transform(teste_x)
classifier.fit(treino_rfecv, treino_y)

matriz_confusao = confusion_matrix(teste_y, classifier.predict(teste_rfecv))

plt.figure(figsize = (10, 8))
sns.set(font_scale= 3)
sns.heatmap(matriz_confusao, 
            annot= True,
            fmt= "d").set(xlabel = "Predição", ylabel = "Real")

print("Resultado da Classificação é %.2f%%" % (classifier.score(teste_rfecv, teste_y) * 100))


# In[65]:


selecionar_rfecv.n_features_


# In[69]:


treino_x.columns[selecionar_rfecv.support_]


# In[70]:


len(selecionar_rfecv.grid_scores_)


# In[77]:


plt.figure(figsize = (14, 8))
plt.xlabel("Nº de Exames")
plt.ylabel("Accuracy")
plt.grid()
plt.plot(range(1, len(selecionar_rfecv.grid_scores_) +1), selecionar_rfecv.grid_scores_ )
plt.show()


# In[79]:


resultados_exames.head()


# In[80]:


selecionar_rfe = RFE(estimator= classifier, n_features_to_select=2, step=1)
selecionar_rfe.fit(treino_x, treino_y)


# In[82]:


valores_exames_v7 = selecionar_rfe.transform(valores_exames_v6)
valores_exames_v7.shape


# In[84]:


plt.figure(figsize = (14, 8))
sns.scatterplot(x= valores_exames_v7[:,0], y= valores_exames_v7[:,1], hue= diagnostico)


# ## Técnica PCA 

# In[85]:


from sklearn.decomposition import PCA


# In[97]:


pca = PCA(n_components= 2)
valores_exames_v8 = pca.fit_transform(valores_exames_v5)


# In[98]:


valores_exames_v8


# In[100]:


plt.figure(figsize = (14, 8))
sns.scatterplot(x= valores_exames_v8[:,0], y= valores_exames_v8[:,1], hue= diagnostico)


# ## Técnica TSNE

# In[107]:


from sklearn.manifold import TSNE


# In[106]:


tsne = TSNE(n_components= 2)
valores_exames_v9 = tsne.fit_transform(valores_exames_v5)
plt.figure(figsize = (14, 8))
sns.scatterplot(x= valores_exames_v9[:,0], y= valores_exames_v9[:,1], hue= diagnostico)

