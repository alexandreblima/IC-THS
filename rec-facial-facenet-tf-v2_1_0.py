# arquivo: rec-facial-facenet-tf-v2_1_0.py 
# Reconhecimento facial com a arquitetura FaceNet
# código original de Antonio José G. Busson et al
# artigo: "Desenvolvendo Modelos de Deep Learning para Aplicações Multimídia 
# no Tensorflow"
# XXIV Simpósio Brasileiro de Sistemas Multimídia e Web (2018): Minicursos

# Código para TensorFlow 2
# O código para TensorFlow 1.14 foi convertido para TensorFlow 2.1.0 por meio 
# do comando a seguir digitado no shell (terminal):
# >> tf_upgrade_v2 --infile rec-facial-facenet-tf-v1_14.py 
# --outfile rec-facial-facenet-tf-v2_1_0.py
# certifique-se que vc está no diretório correto

# A seguinte alteração foi realizada na linha 408 do módulo facenet.py
# graph_def = tf.GraphDef() ==> linha foi comentada com # e substituída pela 
# linha abaixo:
# graph_def = tf.compat.v1.GraphDef() # para viabilizar a conversão de 
# código TF 1.x para TF 2.x

import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import numpy as np

# arquivo local com a arquitetura facenet
# linha de código modificada
import facenet.src.facenet

import cv2 # opencv biblioteca para processamento de imagem
# Linux Ubuntu 18.04: instalar opencv pelo terminal:
# $~ pip install opencv-python==3.4.2.17    
# $~ pip install opencv-contrib-python==3.4.2.17
# em que $~ representa o prompt ... 

# zerar variáveis
from IPython import get_ipython
get_ipython().magic('reset -sf') 

# modelo pré-treinado do FaceNet 

# inicia a sessão no Tensorflow 
sess = tf.compat.v1.Session() # 


# Carregando do modelo pré−treinado
# linha de código modificada 
facenet.src.facenet.load_model('datasets/facematch/20170512-110547.pb')

# seleção dos tensores necessários para obter os embeddings das imagens faciais

# Selecionando os tensores necessarios para execucao
image_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0") 
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0") 
train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

# imprimindo as informações dos tensores
print(image_placeholder)
print(embeddings)
print(train_placeholder)

# A função abaixo utiliza os tensores do FaceNet carregados para obters os embeddings 
# de uma imagem facial. Primeiro, o arquivo de imagem é aberto e redimensionado para o 
# padrão aceito pelo FaceNet 160x160. Em seguida, a imagem é colocada como entrada e o 
# placeholder de treinamento é setado como False. O embedding retornado da execução do 
# grafo de computação é retornado, bem como a imagem que foi usada como entrada.

def get_embedding(img_path): 
    img_size = 160
    img = cv2.imread(img_path)
    #o opencv abre a imagem em BGR, necessario converter para RGB
    if img is None:
        print("Imagem não pode ser aberta.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Preparando a imagem de entrada
    resized = cv2.resize(img, (img_size,img_size),interpolation=cv2.INTER_CUBIC)
    reshaped = resized.reshape(-1,img_size, img_size,3)
    #Configurando entrada e execucao do FaceNet
    feed_dict = {image_placeholder: reshaped, train_placeholder: False}
    embedding = sess.run(embeddings , feed_dict=feed_dict) 
    return embedding[0], img

# Registrando pessoas
# A tarefa de reconhecimento facial tenta responder a pergunta "Quem é essa pessoa?". 
# Para isso, é necessário registrar os embeddings de imagens faciais, para que dessa forma, 
# seja possível realizar a comparação de similaridade entre as imagens faciais. Neste exemplo, 
# são usadas imagens de algumas figuras públicas.    
    
database = {}

database["jennifer"], img = get_embedding("faces/jennifer_0.png")
print("Jennifer Aniston - foto cadastrada:")
_ = plt.imshow(img)
plt.pause(0.1)

database["jolie"], img = get_embedding("faces/jolie_0.png")
print("Angelina Jolie - foto cadastrada:")
_ = plt.imshow(img)
plt.pause(0.1)

database["ozzy"], img = get_embedding("faces/ozzy_0.png")
print("Ozzy Osbourne - foto cadastrada:")
_ = plt.imshow(img)
plt.pause(0.1)

database["brad"], img = get_embedding("faces/brad_0.png")
print("Brad Pitt - foto cadastrada:")
_ = plt.imshow(img)
plt.pause(0.1)

#Este é um exemplo de embedding
print("Facial Embedding da Jennifer:\n", database["jennifer"])

# Reconhecimento facial

# Nesta etapa é realizado o processo de reconhecimento facial. Para isso, como dito
# anteriormente, é calculada a similaridade entre os embeddings das imagens faciais. 
# Uma forma simples para calcular essa similaridade é usando a equação da distancia euclidiana,
# como mostra a função abaixo.

# Função que calcula a distancia euclidiana entre dois vetores
def distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2)**2))

# A função "who_is_it" definida abaixo, identifica uma imagem facial. A função recebe 
# como parâmetro o caminho de um arquivo de imagem e o dicionário de pessoas registradas. 
# Resumidamente, essa função calcula a distancia euclidiana entre os embeddings da imagem de
# entrada e das pessoas registradas, o menor distancia é atribuida a identidade da pessoa.
    
def who_is_it(visitor_image_path, database):
    min_dist = 1000 
    identity = -1
    #Calculando o embedding do visitante
    visitor, img = get_embedding(visitor_image_path)
    #Calculando a distacia do visitante com os demais funcionarios
    for name, employee in database.items():
        dist = distance(visitor, employee)
        
        if dist < min_dist:
            min_dist = dist 
            identity = name
    #verificando a identidade
    if min_dist > 0.5:
        print("Essa pessoa nao esta cadastrada!")
        return None, img
    else:
        return identity, img
    
# Realizando testes    

# Na última etapa, a função "who_is_it" é utilizada para identificar uma imagem facial. 
# No código abaixo são apresentados quatro exemplos de identificação.

identity, img = who_is_it("faces/jennifer_1.png", database)
print("Essa pessoa é o(a)",identity,"!")
# if identity == "jennifer":
# print("Hi, I am Jennifer!")
_ = plt.imshow(img)
plt.pause(0.1)

identity, img = who_is_it("faces/jolie_1.png", database)
print("Essa pessoa é o(a)",identity,"!")
_ = plt.imshow(img)
plt.pause(0.1)

identity, img = who_is_it("faces/ozzy_1.png", database)
print("Essa pessoa é o(a)",identity,"!")
_ = plt.imshow(img)
plt.pause(0.1)

identity, img = who_is_it("faces/brad_1.png", database)
print("Essa pessoa é o(a)",identity,"!")
_ = plt.imshow(img)
plt.pause(0.1)












    




