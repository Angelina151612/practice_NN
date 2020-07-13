import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/SS_tot)

n1=7
n2=5
optsStr=['SGD','Nadam','RMSprop'] 
num_model =29
opt = 0
speed = 0.0003
max_epoch = 10000
step = 200
best_epoch=1927

if not os.path.exists(r'weight/test'):
                   os.mkdir(r'weight/test')
if not os.path.exists(r'weight/test/' + str(num_model)):
                   os.mkdir(r'weight/test/'+ str(num_model))

# загрузка лучших весов
W = load_model(r'weight/BPopt_'+ optsStr[opt]+'/№' + str(n1)+'_'+str(n2)+'_'+str(speed)+'/' + str(num_model)+'/'+'BP_'+ str(n1)+'_'+str(n2)+'_' + str(num_model)+'.h5', custom_objects={'r_square': r_square})
w_0,w_0b= W.layers[0].get_weights()
w_1,w_1b= W.layers[1].get_weights()
w_2,w_2b= W.layers[2].get_weights()
#np.savetxt('weight/test/'+ str(num_model)+'/weight_'+str(num_model)+'_end.csv', np.round(w_0,2),fmt = '% 1.2f', delimiter=",")


# загрузка лучших весов
Wi = load_model(r'weight/BPopt_'+ optsStr[opt]+'/№' + str(n1)+'_'+str(n2)+'_'+str(speed)+'/' + str(num_model)+'/'+'initBP_'+ str(n1)+'_'+str(n2)+'_' + str(num_model)+'.h5', custom_objects={'r_square': r_square})
w_i0,w_i0b= Wi.layers[0].get_weights()
w_i1,w_i1b= Wi.layers[1].get_weights()
w_i2,w_i2b= Wi.layers[2].get_weights()
#np.savetxt('weight/test/'+ str(num_model)+'/weight_'+str(num_model)+'_init.csv', np.round(w_i0,2), fmt = '% 1.2f',delimiter=",")


def weight_inf(w,w_i,name,l1,l2,t):
    print('Разница в весах '+ name)
    print(np.round(w-w_i,2))
    print()
    # гистограмма начальных весов
    w_i = pd.Series(w_i.flatten())
    ax_begin = w_i.hist(bins=20)  
    ax_begin.set_title('Архитектура 7_5, модель ' +  str(num_model)+', начальное распределение весов \n' + name)
    plt.savefig('weight/test/'+ str(num_model)+'/hist_init_'+str(l1)+'_'+str(l2)+'_'+t+'.png',bbox_inches='tight')
    plt.show()
    
    # гистограмма конечных весов
    w = pd.Series(w.flatten())
    ax_end = w.hist(bins=20)  
    ax_end.set_title('Архитектура 7_5, модель ' +  str(num_model)+', конечное распределение весов \n' + name)
    plt.savefig('weight/test/'+ str(num_model)+'/hist_end_'+str(l1)+'_'+str(l2)+'_'+t+'.png',bbox_inches='tight')
    plt.show()

name_0 = 'между входным и первым скрытым cлоем'
#weight_inf(w_0, w_i0, name_0,l1=0, l2=1, t='')

name_b0 = 'нейрона смещения между входным и первым скрытым слоем'
#weight_inf(w_0b, w_i0b, name_b0,l1=0, l2=1, t='bias')

name_1 = 'между первым и вторым скрытым cлоем'
#weight_inf(w_1, w_i1, name_1,l1=1, l2=2, t='')

name_b1 = 'нейрона смещения между первым и вторым скрытым слоем'
#weight_inf(w_1b, w_i1b, name_b1,l1=1, l2=2, t='bias')

name_2 = 'между вторым скрытым и выходным слоем'
#weight_inf(w_2, w_i2, name_2,l1=2, l2=3, t='')

name_b2 = 'нейрона смещения между вторым скрытым и выходным слоем'
#weight_inf(w_2b, w_i2b, name_b2,l1=2, l2=3, t='bias')


# нужные эпохи для вывода
s = range(0,max_epoch,step)


def get_weight(num_epoch):
    W = load_model(r'weight/BPopt_'+ optsStr[opt]+'/№' + str(n1)+'_'+str(n2)+'_'+str(speed)+'/' + str(num_model)+'/'+'checkBP_'+ str(n1)+'_'+str(n2)+'_' + str(num_model)+'_'+str(num_epoch)+'.h5', custom_objects={'r_square': r_square})
    w_0,w_0b= W.layers[0].get_weights()
    w_1,w_1b= W.layers[1].get_weights()
    w_2,w_2b= W.layers[2].get_weights()
    return(w_0,w_0b,w_1,w_1b,w_2,w_2b)

# индексы весов, изменение которых мы хотим отследить
ind_0 = [[0,1],[5,3],[6,3],[5,4],[8,3]]
ind_0b = [0,4]

ind_1 = [[0,2],[2,0],[2,2]]
ind_1b = [0,3]

ind_2 = [[0,0],[2,0]]
ind_2b = [0]


# веса между входным и скрытым
weight_change_0 = []
for i in range(len(ind_0)):
    weight_change_0.append([])
    
# веса смещения между входным и скрытым    
weight_change_0b = []
for i in range(len(ind_0b)):
    weight_change_0b.append([])

# веса между 1 и 2 скрытым
weight_change_1 = []
for i in range(len(ind_1)):
    weight_change_1.append([])
    
# веса смещения между 1 и 2 скрытым    
weight_change_1b = []
for i in range(len(ind_1b)):
    weight_change_1b.append([])
    
# веса между 2 скрытым и выходным
weight_change_2 = []
for i in range(len(ind_2)):
    weight_change_2.append([])
    
# веса смещения между 2 скрытым и выходным    
weight_change_2b = []


for i in s:
    print(i)
    w_0,w_0b,w_1,w_1b,w_2,w_2b = get_weight(i)
    for i in range(len(ind_0)):
        weight_change_0[i].append(w_0[ind_0[i][0],ind_0[i][1]])
    for i in range(len(ind_0b)):
        weight_change_0b[i].append(w_0b[ind_0b[i]])
        
    for i in range(len(ind_1)):
        weight_change_1[i].append(w_1[ind_1[i][0],ind_1[i][1]])
    for i in range(len(ind_1b)):
        weight_change_1b[i].append(w_1b[ind_1b[i]])
        
    for i in range(len(ind_2)):
        weight_change_2[i].append(w_2[ind_2[i][0],ind_2[i][1]])
    weight_change_2b.append(w_2b)

    
def print_weight(i,j,weights,name,opt,t,l1,l2,index):
    if(t=='neur'):
        plt.title('Изменение веса между '+ str(i+1)+ ' и ' + str(j+1)+ ' нейронами \n' +name+'\n'+ optsStr[opt])
        plt.plot(s,weights, 'ro', color = 'orangered')
    else:
        plt.title('Изменение веса '+name+'\n'+ optsStr[opt])
        plt.plot(s,weights, 'ro', color = 'green')
    plt.plot([best_epoch,best_epoch],[min(weights),max(weights)], color = 'blue')  
    plt.ylabel('веса')
    plt.xlabel('эпоха')
    if(t=='neur'):
        plt.savefig('weight/test/'+ str(num_model)+'/weight_change_'+str(l1)+'_'+str(l2)+'_'+str(index)+'_'+optsStr[opt]+'.png',bbox_inches='tight')
    else:
        plt.savefig('weight/test/'+ str(num_model)+'/weight_change_'+ str(l1)+'_'+str(l2)+'_'+str(index)+'_bias'+'_'+optsStr[opt]+'.png',bbox_inches='tight')
    plt.show()

for i in range(len(ind_0)):
    print_weight(ind_0[i][0], ind_0[i][1],np.array(weight_change_0[i]),name_0,opt,'neur',0,1,i)
    
for i in range(len(ind_0b)):
    print_weight(ind_0b[i],ind_0b[i],np.array(weight_change_0b[i]),name_b0,opt,'bias',0,1,i)
    
for i in range(len(ind_1)):
    print_weight(ind_1[i][0], ind_1[i][1],np.array(weight_change_1[i]),name_1,opt,'neur',1,2,i)

for i in range(len(ind_1b)):
    print_weight(ind_1b[i],ind_1b[i],np.array(weight_change_1b[i]),name_b1,opt,'bias',1,2,i)

for i in range(len(ind_2)):
    print_weight(ind_2[i][0], ind_2[i][1],np.array(weight_change_2[i]),name_2,opt,'neur',2,3,i)


print_weight(ind_2b,ind_2b,np.array(weight_change_2b),name_b2,opt,'bias',2,3,0)
