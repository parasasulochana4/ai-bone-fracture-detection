import torch, torchvision, glob, numpy as np
from PIL import Image
torch.set_num_threads(2)
m = torchvision.models.densenet121(weights="IMAGENET1K_V1").eval()
fs = sorted(glob.glob("raw/*.png"))
labels = np.array([int(f[-5]) for f in fs])
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
def load(f):
    im = Image.open(f).convert("RGB").resize((224,224), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im)).permute(2,0,1).float()/255
mid = np.zeros((len(fs),512,7,7), np.float16); gap = np.zeros((len(fs),1024), np.float32)
f = m.features
pre = torch.nn.Sequential(*list(f.children())[:-2])  # up to transition3 (+pool) -> 512x7x7
post = torch.nn.Sequential(f.denseblock4, f.norm5)
with torch.no_grad():
    for i in range(0,len(fs),32):
        x = torch.stack([load(p) for p in fs[i:i+32]]); x=(x-mean)/std
        a = pre(x); b = torch.relu(post(a)).mean((2,3))
        mid[i:i+32]=a.numpy().astype(np.float16); gap[i:i+32]=b.numpy()
        if i%640==0: print(i, flush=True)
np.save("mid.npy",mid); np.save("gap.npy",gap); np.save("labels.npy",labels)
open("files.txt","w").write("\n".join(fs)); print("done")
