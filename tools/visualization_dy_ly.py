import numpy as np
import matplotlib.pyplot as plt

folder_lists=['results/loss_adjust_ly_pretrain']#,'results/ly_only_loss_cifar10_1']
results=[]
acc_lists=[]
read_epoch=400
num_classes=10
for folder in folder_lists:
    results.append([])
    acc_lists.append([])
    acc_file=open(f'{folder}/acc.txt',mode='r')
    temp_acc=[]
    for lines in acc_file:
        temp_acc.append([100.-float(x) for x in lines.split(' ')])
    temp_acc=temp_acc[:read_epoch]
    acc_lists[-1].append(temp_acc)

    dy_file=open(f'{folder}/dy.txt',mode='r')
    temp_dy=[]
    for lines in dy_file:
        if not temp_dy or len(temp_dy[-1])==num_classes:
            temp_dy.append([float(x) for x in lines.replace('[','').replace(']','').replace('\n','').split()])
        else:
            temp_dy[-1].extend([float(x) for x in lines.replace('[','').replace(']','').replace('\n','').split()])
    #print(temp_dy)
    temp_dy=temp_dy[:read_epoch]
    results[-1].append(temp_dy)

    ly_file=open(f'{folder}/ly.txt',mode='r')
    temp_ly=[]
    for lines in ly_file:
        if not temp_ly or len(temp_ly[-1])==num_classes:
            temp_ly.append([float(x) for x in lines.replace('[','').replace(']','').replace('\n','').split()])
        else:
            temp_ly[-1].extend([float(x) for x in lines.replace('[','').replace(']','').replace('\n','').split()])
    temp_ly=temp_ly[:read_epoch]
    results[-1].append(temp_ly)
results=np.array(results)
acc_lists=np.array(acc_lists)
print(results.shape)
print(acc_lists.shape)

# draw
colors=['tab:blue','tab:green','tab:red','tab:brown','tab:orange']
legends=['01','23','45','67','89']
for folder in folder_lists:
    i=folder_lists.index(folder)
    plt.cla()
    for j in range(num_classes//2):
        plt.plot(np.mean(results[i,0,:,2*j:2*j+1],axis=1),color=colors[j],linewidth=3)
    plt.legend(legends)
    for j in range(num_classes//2):
        plt.plot(np.mean(results[i,1,:,2*j:2*j+1],axis=1),color=colors[j],linewidth=3,linestyle='--')
    
    plt.grid('--')
    plt.savefig(f'{folder}/fig_dyly.pdf')
    #plt.show()

colors=['tab:blue','tab:green','tab:red','tab:brown','tab:orange']
legends=['train','val','test']
for folder in folder_lists:
    i=folder_lists.index(folder)
    plt.cla()
    for j in range(3):
        #print(acc_lists)
        plt.plot(acc_lists[i,0,:,j],color=colors[j],linewidth=3)
    plt.legend(legends)
    plt.ylim((-5,100))
    plt.grid('--')
    plt.savefig(f'{folder}/fig_acc.pdf')

