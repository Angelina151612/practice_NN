import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/SS_tot)


n1=7
n2=5
optsStr=['SGD','Nadam','RMSprop'] 
num_model =13
opt = 1
speed = 0.0003

# загрузка лучших весов
W = load_model(r'weight/BPopt_'+ optsStr[opt]+'/№' + str(n1)+'_'+str(n2)+'_'+str(speed)+'/' + str(num_model)+'/'+'BP_'+ str(n1)+'_'+str(n2)+'_' + str(num_model)+'.h5', custom_objects={'r_square': r_square})
w_0,w_0b= W.layers[0].get_weights()
#np.savetxt("weight/BPopt_Nadam/weight_13_end.csv", np.round(w_0,2),fmt = '% 1.2f', delimiter=",")

# загрузка лучших весов
Wi = load_model(r'weight/BPopt_'+ optsStr[opt]+'/№' + str(n1)+'_'+str(n2)+'_'+str(speed)+'/' + str(num_model)+'/'+'initBP_'+ str(n1)+'_'+str(n2)+'_' + str(num_model)+'.h5', custom_objects={'r_square': r_square})
w_i0,w_i0b= Wi.layers[0].get_weights()
#np.savetxt("weight/BPopt_Nadam/weight_13_init.csv", np.round(w_i0,2), fmt = '% 1.2f',delimiter=",")

# разница весов
#print(np.round(w_0-w_i0,2))


# гистограмма begin
w_i0 = pd.Series(w_i0.flatten())
ax_end = w_i0.hist(bins=20)  
ax_end.set_title('Архитектура 7_5, модель ' +  str(num_model)+', начальное распределение весов')
plt.show()

#г истограмма end
w_0 = pd.Series(w_0.flatten())
ax_end = w_0.hist(bins=20)  
ax_end.set_title('Архитектура 7_5, модель ' +  str(num_model)+', конечное распределение весов')
plt.show()


#функция для получения весов на конкретных эпохах
def get_weight(num_epoch):
    W = load_model(r'weight/BPopt_'+ optsStr[opt]+'/№' + str(n1)+'_'+str(n2)+'_'+str(speed)+'/' + str(num_model)+'/'+'checkBP_'+ str(n1)+'_'+str(n2)+'_' + str(num_model)+'_'+str(num_epoch)+'.h5', custom_objects={'r_square': r_square})
    w_0,w_0b= W.layers[0].get_weights()
    return(w_0)

s = range(0,10000,200)
ind1 = 8
ind11 = 1

ind2 = 9
ind22 = 2

ind3 = 1
ind33 = 2

ind4 = 5
ind44 = 6

ind5 = 0
ind55 = 3

weight_change_1 = []
weight_change_2 = []
weight_change_3 = []
weight_change_4 = []
weight_change_5 = []

for i in s:
    print(i)
    w = get_weight(i)
    weight_change_1.append(w[ind1,ind11])
    weight_change_2.append(w[ind2,ind22])
    weight_change_3.append(w[ind3,ind33])
    weight_change_4.append(w[ind4,ind44])
    weight_change_5.append(w[ind5,ind55])

# лучшая эпоха
x=375

def print_weight(i,j,weights):
    plt.title('Изменение веса между '+ str(i+1)+ ' и ' + str(j+1)+ ' нейронами')
    plt.plot([x,x],[min(weights),max(weights)])
    plt.plot(s,weights, 'ro')
    plt.ylabel('веса')
    plt.xlabel('эпоха')
    plt.show()

print_weight(ind1, ind11,weight_change_1)
print_weight(ind2, ind22,weight_change_2)
print_weight(ind3, ind33,weight_change_3)
print_weight(ind4, ind44,weight_change_4)
print_weight(ind5, ind55,weight_change_5)
