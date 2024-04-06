from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np 


def cal_info(data,mode):

    label=data[:,-1]

    amounts=np.unique(label,return_counts=True)[1]

    prob=amounts/amounts.sum()

    if mode=='info':
        res=(np.log2(prob)*(-prob)).sum()
    else:
        res=1-np.power(prob,2).sum()

    return res

    

def split(data,col,value):

    con=data[:,col]<=value

    l_data=data[con]
    r_data=data[~con]

    return l_data,r_data


class Tree:

    def __init__(self,col=-1,value=None,leaf=None,l=None,
                 r=None,mode='info'):
        
        self.col=col
        self.value=value
        self.leaf=leaf
        self.l=l
        self.r=r
        self.mode=mode

def build(data,mode):

    if len(data)==0:
        return Tree(mode=mode)
    
    init_info=cal_info(data,mode)
    diff=0

    m,n=data.shape

    for col in range(n-1):
        for val in data[:,col]:
            l_data,r_data=split(data,col,val)
            new_info=(cal_info(l_data,mode)*len(l_data)+
                      cal_info(r_data,mode)*len(r_data))/len(data)
            
            redu=init_info-new_info

            if redu>diff and len(l_data)>0 and len(r_data)>0:

                mid_l=l_data
                mid_r=r_data
                mid_col=col
                mid_val=val
                diff=redu
    if diff>0:

        l=build(mid_l,mode)
        r=build(mid_r,mode)

        return Tree(col=mid_col,value=mid_val,l=l,r=r,mode=mode)

    else:

        clx,num=np.unique(data[:,-1],return_counts=True)
        d=dict(zip(clx,num))
        leaf=max(d,key=d.get)
        return Tree(leaf=leaf,mode=mode)
        
def pf(tree,level='o'):

    if tree.leaf is not None:
        print(level+'*'+str(tree.leaf))
    else:
        print(level+'-'+str(tree.col)+'-'+str(tree.value))
        pf(tree.l,level+'L ')
        pf(tree.r,level+'R ')


def predict(data,tree):

    if tree.leaf is not None:
        return tree.leaf
    
    else:
        if data[tree.col]<=tree.value:
            branch=tree.l
        else:
            branch=tree.r

        return predict(data,branch)


        
if __name__ == '__main__':
    
    com=load_iris()

    data,label=com['data'],com['target']

    train_x,test_x,train_y,test_y=train_test_split(data,label,shuffle=True)

    trains=np.c_[train_x,train_y]

    tree=build(trains,'info')

    pf(tree)

    pred_train=np.array([predict(var,tree) for var in train_x])

    score_train=(pred_train==train_y).sum()/len(pred_train)

    print(f'acc_train: {score_train}')

    pred_test=np.array([predict(var,tree) for var in test_x])

    score_test=(pred_test==test_y).sum()/len(test_y)

    print(f'acc_test: {score_test}')
