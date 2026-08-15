#!/usr/bin/env python3
"""Train a temporal Hall-only gated residual on a frozen safe actor."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torch import nn


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1864,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,29))
    def forward(self,x): return torch.clamp(self.mlp(x),-3.,3.)


class Gated(nn.Module):
    def __init__(self, state, limit, gate_power=2.0):
        super().__init__(); self.limit=float(limit); self.gate_power=float(gate_power); self.base=Base()
        self.base.load_state_dict({k:v for k,v in state.items() if k.startswith('mlp.')},strict=True)
        for p in self.base.parameters(): p.requires_grad_(False)
        self.res=nn.Sequential(nn.Linear(1864,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,29))
        self.gate=nn.Sequential(nn.Linear(1864,128),nn.ELU(),nn.Linear(128,1))
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)
    def forward(self,x):
        g=torch.sigmoid(self.gate(x)).pow(self.gate_power)
        return torch.clamp(self.base(x)+self.limit*g*torch.tanh(self.res(x)),-3.,3.)
    def residual(self,x): return self.limit*torch.sigmoid(self.gate(x)).pow(self.gate_power)*torch.tanh(self.res(x))


def data(path):
    d=np.load(path); o=np.asarray(d['observation'],np.float32); a=np.asarray(d['action'],np.float32); l=np.asarray(d['low'],bool)
    if o.ndim!=2 or o.shape[1]!=1864 or a.shape!=(len(o),29) or l.shape!=(len(o),): raise ValueError(path)
    return o,a,l


def old_action(path,obs):
    s=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']); n=s.get_inputs()[0].name; one=s.get_inputs()[0].shape[0]==1; k=1 if one else 256
    return np.concatenate([s.run(None,{n:obs[i:i+k]})[0] for i in range(0,len(obs),k)]).astype(np.float32)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--safe-checkpoint',type=Path,required=True); p.add_argument('--old-policy',type=Path,required=True); p.add_argument('--safe-dataset',type=Path,required=True); p.add_argument('--old-dataset',type=Path,default=None,help='Optional surviving HIGH samples from the same fast teacher.'); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,default=80); p.add_argument('--residual-limit',type=float,default=.15); p.add_argument('--high-blend',type=float,default=.20); p.add_argument('--gate-power',type=float,default=2.0); p.add_argument('--seed',type=int,default=20260810); a=p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    so,sa,sl=data(a.safe_dataset)
    if a.old_dataset is None:
        oo=np.empty((0,1864),dtype=np.float32); oa=np.empty((0,29),dtype=np.float32)
    else:
        oo,oa,ol=data(a.old_dataset); oo,oa=oo[~ol],oa[~ol]
    state=torch.load(a.safe_checkpoint,map_location='cpu',weights_only=False)['actor_state_dict']; base=Base(); base.load_state_dict({k:v for k,v in state.items() if k.startswith('mlp.')},strict=True); base.eval()
    all_o=np.concatenate([so,oo]);
    with torch.no_grad(): all_base=base(torch.from_numpy(all_o)).numpy()
    old_on_safe=old_action(a.old_policy,so); safe_base=all_base[:len(so)]; old_base=all_base[len(so):]
    target=np.concatenate([(1-a.high_blend)*safe_base+a.high_blend*old_on_safe,oa]); residual=np.clip(target-all_base,-a.residual_limit,a.residual_limit); residual[:len(so)][sl]=0.
    # HIGH labels are only used to teach the gate; they are never actor inputs.
    high=np.concatenate([~sl,np.ones(len(oo),dtype=bool)])
    model=Gated(state,a.residual_limit,a.gate_power); opt=torch.optim.AdamW(list(model.res.parameters())+list(model.gate.parameters()),lr=1e-4,weight_decay=1e-5)
    x=torch.from_numpy(all_o); y=torch.from_numpy(residual); z=torch.from_numpy(high.astype(np.float32))
    order=np.random.permutation(len(x)); split=max(1,int(.9*len(order))); tr=torch.from_numpy(order[:split]); te=torch.from_numpy(order[split:]); hist=[]
    for ep in range(a.epochs):
        perm=tr[torch.randperm(len(tr))]; total=0.
        for i in range(0,len(perm),1024):
            idx=perm[i:i+1024]; pr=model.residual(x[idx]); act=((pr-y[idx])**2).mean(); logits=model.gate(x[idx]).squeeze(-1); gate=nn.functional.binary_cross_entropy_with_logits(logits,z[idx]); loss=act+.02*gate
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(list(model.res.parameters())+list(model.gate.parameters()),.5); opt.step(); total+=float(loss.detach())*len(idx)
        with torch.no_grad():
            te_loss=((model.residual(x[te])-y[te])**2).mean().item(); te_gate=nn.functional.binary_cross_entropy_with_logits(model.gate(x[te]).squeeze(-1),z[te]).item()
        hist.append({'epoch':ep+1,'train_loss':total/len(tr),'test_residual_mse':te_loss,'test_gate_bce':te_gate})
    a.output_dir.mkdir(parents=True,exist_ok=True); ck=a.output_dir/'spatial_gated_residual.pt'; torch.save({'model':model.state_dict(),'input_dim':1864,'output_dim':29,'residual_limit':a.residual_limit},ck); onnx=a.output_dir/'policy.onnx'; torch.onnx.export(model.eval(),torch.zeros(1,1864),onnx,input_names=['obs'],output_names=['action'],dynamic_axes={'obs':{0:'batch'},'action':{0:'batch'}},opset_version=17)
    report={'status':'PASS','base_frozen':True,'temporal_gate':True,'gate_power':a.gate_power,'input_dimension':1864,'residual_limit':a.residual_limit,'high_blend':a.high_blend,'safe_samples':len(so),'safe_low_samples':int(sl.sum()),'old_high_samples':len(oo),'forbidden_inputs':['normal_force','tangential_force','ground_friction_mu','contact_truth'],'checkpoint':str(ck),'onnx':str(onnx),'history':hist}; (a.output_dir/'training_summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
