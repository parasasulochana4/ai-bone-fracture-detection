# Frozen-backbone baselines (ImageNet features + logistic regression) on the same leak-free split.
import torch, torchvision, numpy as np, json
from PIL import Image
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2)
fs=open("files.txt").read().split("\n"); lab=np.load("labels.npy"); y=(lab==0).astype(int)
sp=np.load("split.npz"); tr,te=sp["train"],sp["test"]
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
def load(p): return torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((224,224),Image.BILINEAR))).permute(2,0,1).float()/255
res={}
for name,ctor,w in [("ResNet50",torchvision.models.resnet50,"IMAGENET1K_V2"),("EfficientNet-B0",torchvision.models.efficientnet_b0,"IMAGENET1K_V1")]:
    m=ctor(weights=w).eval()
    if name=="ResNet50": m.fc=torch.nn.Identity()
    else: m.classifier=torch.nn.Identity()
    feats=[]
    with torch.no_grad():
        for i in range(0,len(fs),32):
            feats.append(m((torch.stack([load(p) for p in fs[i:i+32]])-mean)/std).numpy())
    X=np.concatenate(feats); np.save(f"feat_{name}.npy",X)
    acc=LogisticRegression(C=0.5,max_iter=3000).fit(X[tr],y[tr]).score(X[te],y[te]); res[name]=acc; print(name,acc,flush=True)
gap=np.load("gap.npy"); res["DenseNet121 (frozen)"]=LogisticRegression(C=0.5,max_iter=3000).fit(gap[tr],y[tr]).score(gap[te],y[te])
json.dump(res,open("baselines.json","w"),indent=1); print(res)
