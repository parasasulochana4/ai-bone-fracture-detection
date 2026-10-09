# Cache DenseNet121 transition-2 outputs (256x14x14) so denseblock3+4 can be fine-tuned on CPU.
import torch, torchvision, numpy as np
from PIL import Image
torch.set_num_threads(2)
m = torchvision.models.densenet121(weights="IMAGENET1K_V1").eval()
fs = open("files.txt").read().split("\n")
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
f = m.features
pre = torch.nn.Sequential(f.conv0,f.norm0,f.relu0,f.pool0,f.denseblock1,f.transition1,f.denseblock2,f.transition2)
def load(p, flip):
    im = Image.open(p).convert("RGB").resize((224,224), Image.BILINEAR)
    if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return torch.from_numpy(np.array(im)).permute(2,0,1).float()/255
out = np.lib.format.open_memmap("mid2.npy", mode="w+", dtype=np.float16, shape=(len(fs),256,14,14))
with torch.no_grad():
    for i in range(0,len(fs),32):
        x = torch.stack([load(p, False) for p in fs[i:i+32]]); x=(x-mean)/std
        out[i:i+32] = pre(x).numpy().astype(np.float16)
        if i%1280==0: print(i, flush=True)
out.flush(); print("done")
