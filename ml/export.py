# Export the fine-tuned DenseNet121 (with flip test-time augmentation) to ONNX and evaluate the
# exported model end-to-end on the held-out test images. The resulting metrics.json is what the site shows.
import torch, torchvision, numpy as np, json, copy, sys, onnxruntime as ort
from PIL import Image
variant=sys.argv[1]  # "block4" (head_best.pt) or "block3" (head2_best.pt)
base=torchvision.models.densenet121(weights="IMAGENET1K_V1").eval(); f=base.features
class Full(torch.nn.Module):
    def __init__(s):
        super().__init__()
        if variant=="block3":
            s.pre=torch.nn.Sequential(f.conv0,f.norm0,f.relu0,f.pool0,f.denseblock1,f.transition1,f.denseblock2,f.transition2)
            s.body=torch.nn.Sequential(copy.deepcopy(f.denseblock3),copy.deepcopy(f.transition3),copy.deepcopy(f.denseblock4),copy.deepcopy(f.norm5))
            sd=torch.load("head2_best.pt"); s.body.load_state_dict({k[5:]:v for k,v in sd.items() if k.startswith("body.")})
        else:
            s.pre=torch.nn.Sequential(*list(f.children())[:-2])
            s.body=torch.nn.Sequential(copy.deepcopy(f.denseblock4),copy.deepcopy(f.norm5))
            sd=torch.load("head_best.pt"); s.body.load_state_dict({("0."+k[6:] if k.startswith("block.") else "1."+k[5:]):v for k,v in sd.items() if k.startswith(("block.","norm."))})
        s.fc=torch.nn.Linear(1024,1); s.fc.load_state_dict({"weight":sd["fc.weight"],"bias":sd["fc.bias"]})
    def one(s,x):
        feat=torch.relu(s.body(s.pre(x)))
        logit=s.fc(feat.mean((2,3))).squeeze(1)
        cam=torch.einsum("c,bchw->bhw",s.fc.weight[0],feat)+s.fc.bias
        return torch.sigmoid(logit),cam
    def forward(s,image):
        p1,c1=s.one(image); p2,c2=s.one(image.flip(3))
        return (p1+p2)/2,(c1+c2.flip(2))/2
m=Full().eval()
dummy=torch.randn(1,3,224,224)
torch.onnx.export(m,dummy,"fracture-densenet121.onnx",input_names=["image"],output_names=["fracture_probability","cam"],opset_version=17,dynamo=False)
sess=ort.InferenceSession("fracture-densenet121.onnx")
fs=open("files.txt").read().split("\n"); lab=np.load("labels.npy"); y=(lab==0).astype(int); te=np.load("split.npz")["test"]
mean=np.array([0.485,0.456,0.406],np.float32).reshape(3,1,1); std=np.array([0.229,0.224,0.225],np.float32).reshape(3,1,1)
probs=[]
for i in te:
    g=np.asarray(Image.open(fs[i]).convert("L").resize((224,224),Image.BILINEAR),np.float32)/255
    x=((np.stack([g]*3)-mean)/std)[None].astype(np.float32)
    probs.append(sess.run(None,{"image":x})[0][0])
with torch.no_grad():
    x=torch.from_numpy(x); assert abs(m(x)[0].item()-probs[-1])<1e-3
pr=(np.array(probs)>=0.5).astype(int); yt=y[te]
tp=int(((pr==1)&(yt==1)).sum()); tn=int(((pr==0)&(yt==0)).sum()); fp=int(((pr==1)&(yt==0)).sum()); fn=int(((pr==0)&(yt==1)).sum())
res=dict(variant=variant,test_accuracy=(tp+tn)/len(te),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp),test_size=len(te),confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn))
print(json.dumps(res,indent=1)); json.dump(res,open("metrics.json","w"),indent=1)
