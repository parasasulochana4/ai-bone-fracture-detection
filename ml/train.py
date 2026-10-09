import numpy as np, torch, torchvision, json, copy, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
gap=np.load("gap.npy"); lab=np.load("labels.npy"); y=(lab==0).astype(np.int64)  # 1 = fractured
n=len(y)
# Group near-duplicates (the source dataset contains augmented copies of the same X-ray).
g=gap/np.linalg.norm(gap,axis=1,keepdims=True)
T=float(sys.argv[1]) if len(sys.argv)>1 else 0.92
rows,cols=[],[]
for i in range(0,n,1000):
    s=g[i:i+1000]@g.T
    r,c=np.where(s>T); rows+=list(r+i); cols+=list(c)
S=g@g.T; np.fill_diagonal(S,-1); nn=S.argmax(1); keep=S.max(1)>0.85
rows+=list(np.where(keep)[0]); cols+=list(nn[keep]); del S
A=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n))
ng,groups=connected_components(A,directed=False)
print("threshold",T,"groups",ng,"largest",np.bincount(groups).max())
# group-aware split 70/15/15
sgk=StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=0)
folds=[te for _,te in sgk.split(gap,y,groups)]
test=folds[0]; val=folds[1]; train=np.concatenate(folds[2:])
assert not set(groups[test])&set(groups[train])
print("split",len(train),len(val),len(test))
np.savez("split.npz",train=train,val=val,test=test,groups=groups)
clf=LogisticRegression(C=0.5,max_iter=3000).fit(gap[train],y[train])
print("frozen DenseNet121 + logistic regression: val %.4f test %.4f"%(clf.score(gap[val],y[val]),clf.score(gap[test],y[test])))
