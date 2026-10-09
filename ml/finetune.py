# Fine-tune DenseNet121's last dense block + classifier on cached transition-3 features.
import numpy as np, torch, torchvision, json, copy, time
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
mid=np.load("mid.npy", mmap_mode="r"); lab=np.load("labels.npy"); y=torch.tensor((lab==0).astype(np.float32))
sp=np.load("split.npz"); tr,va,te=sp["train"],sp["val"],sp["test"]
base=torchvision.models.densenet121(weights="IMAGENET1K_V1")
class Head(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.block=copy.deepcopy(base.features.denseblock4); s.norm=copy.deepcopy(base.features.norm5)
        s.drop=torch.nn.Dropout(0.3); s.fc=torch.nn.Linear(1024,1)
    def forward(s,x):
        f=torch.relu(s.norm(s.block(x))); return s.fc(s.drop(f.mean((2,3)))).squeeze(1)
m=Head()
opt=torch.optim.AdamW([{"params":list(m.block.parameters())+list(m.norm.parameters()),"lr":1e-4},{"params":m.fc.parameters(),"lr":1e-3}],weight_decay=1e-4)
EPOCHS=int(__import__("sys").argv[1]) if len(__import__("sys").argv)>1 else 8
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
def batch(idx,aug=False):
    x=torch.from_numpy(np.stack([mid[i] for i in idx]).astype(np.float32))
    if aug:
        fl=torch.rand(len(idx))<0.5; x[fl]=x[fl].flip(3)
        fv=torch.rand(len(idx))<0.5; x[fv]=x[fv].flip(2)
    return x
def evaluate(idx):
    m.eval(); ps=[]
    with torch.no_grad():
        for i in range(0,len(idx),128):
            x=batch(idx[i:i+128]); ps.append(((torch.sigmoid(m(x))+torch.sigmoid(m(x.flip(3))))/2))
    p=torch.cat(ps); return ((p>0.5).float()==y[idx]).float().mean().item(), p
best=0; bs=64
for ep in range(EPOCHS):
    m.train(); perm=np.random.permutation(tr); t0=time.time(); tot=0
    for i in range(0,len(perm),bs):
        idx=perm[i:i+bs]; loss=torch.nn.functional.binary_cross_entropy_with_logits(m(batch(idx,True)),y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(idx)
    sched.step(); acc,_=evaluate(va)
    print(f"epoch {ep+1} loss {tot/len(tr):.4f} val {acc:.4f} ({time.time()-t0:.0f}s)",flush=True)
    if acc>best: best=acc; torch.save(m.state_dict(),"head_best.pt")
m.load_state_dict(torch.load("head_best.pt"))
va_acc,_=evaluate(va); te_acc,p=evaluate(te)
yt=y[te].numpy(); pr=(p.numpy()>0.5)
tp=int(((pr==1)&(yt==1)).sum()); tn=int(((pr==0)&(yt==0)).sum()); fp=int(((pr==1)&(yt==0)).sum()); fn=int(((pr==0)&(yt==1)).sum())
res=dict(val_accuracy=va_acc,test_accuracy=te_acc,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp),test_size=len(te),train_size=len(tr),val_size=len(va),confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn))
print(json.dumps(res,indent=1)); json.dump(res,open("metrics.json","w"),indent=1)
