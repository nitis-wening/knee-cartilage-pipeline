# train_unet3d_4class.py
"""
UNet3D with 4 class :
  1 = Patellar
  2 = Femoral
  3 = Tibial    (TC-med + TC-lat merged)
  4 = Meniscus  (Men-med + Men-lat merged)

Config same with train_unet3d.py (6 class):
  channels=[32,64,128,256], patch=80³, LR=1e-4, batch=2
"""

import os, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

# Update these paths according to your setup
DATA_ROOT   = '/data1/nitis/kneeproject/data/qdess/v1-release'
ANNOT_DIR   = os.path.join(DATA_ROOT, 'annotations/v1.0.0')
NPY_DIR     = '/data1/nitis/kneeproject/data/qdess_npy_1mm'
CKPT_DIR    = '/data1/nitis/kneeproject/checkpoints'
RESULTS_DIR = '/data1/nitis/kneeproject/results'
LOG_DIR     = '/data1/nitis/kneeproject/logs'

NUM_CLASSES   = 4
IN_CHANNELS   = 2
VOXEL_SPACING = (1.0, 1.0, 1.0)
CORRUPT_FILES = {'MTR_172.h5'}
LABEL_NAMES   = ['Patellar', 'Femoral', 'Tibial', 'Meniscus']

CHANNELS  = [32, 64, 128, 256]
DROPOUT   = 0.1

PATCH_SIZE     = (80, 80, 80)
BATCH_SIZE     = 2
NUM_WORKERS    = 2
POS_RATIO      = 0.8
LR             = 1e-4
LR_GAMMA       = 0.9999
BETAS          = (0.9, 0.99)
MAX_EPOCHS     = 500
VAL_INTERVAL   = 10
SAVE_INTERVAL  = 5
SEED           = 42
EARLY_STOP_PAT = 50
OVERLAP        = 0.5
RESUME         = False

CKPT_PATH = os.path.join(CKPT_DIR, 'unet3d_4class_resume.pt')
BEST_PATH = os.path.join(CKPT_DIR, 'unet3d_4class_best.pt')

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, num_classes=4, channels=[32,64,128,256], dropout=0.1):
        super().__init__()
        self.enc1=ConvBlock(in_channels, channels[0], dropout)
        self.enc2=ConvBlock(channels[0], channels[1], dropout)
        self.enc3=ConvBlock(channels[1], channels[2], dropout)
        self.enc4=ConvBlock(channels[2], channels[3], dropout)
        self.pool=nn.MaxPool3d(2)
        self.bottleneck=ConvBlock(channels[3], channels[3]*2, dropout)
        self.up4=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=False)
        self.dec4=ConvBlock(channels[3]*2+channels[3], channels[3], dropout)
        self.up3=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=False)
        self.dec3=ConvBlock(channels[3]+channels[2], channels[2], dropout)
        self.up2=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=False)
        self.dec2=ConvBlock(channels[2]+channels[1], channels[1], dropout)
        self.up1=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=False)
        self.dec1=ConvBlock(channels[1]+channels[0], channels[0], dropout)
        self.out_conv=nn.Conv3d(channels[0], num_classes+1, 1)

    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.out_conv(d1)

def clip_and_normalize(vol):
    vol=vol.astype(np.float32); mask=vol>0
    if mask.sum()==0: return vol
    nz=vol[mask]; vol=np.clip(vol,np.percentile(nz,0.5),np.percentile(nz,99.5))
    vmin=vol[mask].min(); vmax=vol[mask].max()
    if vmax>vmin: vol=(vol-vmin)/(vmax-vmin)
    vol[~mask]=0.0; return vol.astype(np.float32)

def seg_to_label_4class(seg):
    label=np.zeros(seg.shape[:3],dtype=np.int64)
    label[seg[...,0]]=1  # Patellar
    label[seg[...,1]]=2  # Femoral
    label[seg[...,2]]=3  # TC-med  → Tibial
    label[seg[...,3]]=3  # TC-lat  → Tibial
    label[seg[...,4]]=4  # Men-med → Meniscus
    label[seg[...,5]]=4  # Men-lat → Meniscus
    return label

def augment(image, label):
    if random.random()<0.5:
        image=np.flip(image,axis=2).copy(); label=np.flip(label,axis=1).copy()
    if random.random()<0.3:
        image=np.flip(image,axis=1).copy(); label=np.flip(label,axis=0).copy()
    if random.random()<0.3:
        k=random.randint(1,3)
        image=np.rot90(image,k=k,axes=(1,2)).copy(); label=np.rot90(label,k=k,axes=(0,1)).copy()
    if random.random()<0.3:
        for c in range(image.shape[0]):
            image[c]=np.clip(image[c]*(1.0+random.uniform(-0.1,0.1)),0.0,1.0)
    if random.random()<0.3:
        for c in range(image.shape[0]):
            image[c]=np.clip(image[c]+random.uniform(-0.1,0.1),0.0,1.0)
    if random.random()<0.2:
        for c in range(image.shape[0]):
            image[c]=np.power(image[c],random.uniform(0.8,1.2)).astype(np.float32)
    return image, label

def sample_patch_balanced(image, label, patch_size, pos_ratio=0.8):
    _,H,W,D=image.shape; pH,pW,pD=patch_size
    ph=max(0,pH-H); pw=max(0,pW-W); pd=max(0,pD-D)
    if ph>0 or pw>0 or pd>0:
        image=np.pad(image,((0,0),(ph//2,ph-ph//2),(pw//2,pw-pw//2),(pd//2,pd-pd//2)),constant_values=0)
        label=np.pad(label,((ph//2,ph-ph//2),(pw//2,pw-pw//2),(pd//2,pd-pd//2)),constant_values=0)
        _,H,W,D=image.shape
    if random.random()<pos_ratio:
        sc=[np.argwhere(label==c) for c in range(1,NUM_CLASSES+1) if len(np.argwhere(label==c))>0]
        if sc:
            chosen=random.choice(sc); center=chosen[np.random.randint(len(chosen))]
            ch,cw,cd=center
            h0=int(np.clip(ch-pH//2,0,H-pH)); w0=int(np.clip(cw-pW//2,0,W-pW)); d0=int(np.clip(cd-pD//2,0,D-pD))
            return image[:,h0:h0+pH,w0:w0+pW,d0:d0+pD].copy(),label[h0:h0+pH,w0:w0+pW,d0:d0+pD].copy()
    h0=random.randint(0,max(0,H-pH)); w0=random.randint(0,max(0,W-pW)); d0=random.randint(0,max(0,D-pD))
    return image[:,h0:h0+pH,w0:w0+pW,d0:d0+pD].copy(),label[h0:h0+pH,w0:w0+pW,d0:d0+pD].copy()

class SKMTEADataset(Dataset):
    def __init__(self,split,npy_dir,annot_dir,patch_size=(80,80,80),pos_ratio=0.8,do_augment=True):
        self.split=split; self.npy_dir=npy_dir; self.patch_size=patch_size
        self.pos_ratio=pos_ratio; self.do_augment=do_augment and (split=='train')
        with open(os.path.join(annot_dir,f'{split}.json')) as f: ann=json.load(f)
        self.file_names=[i['file_name'] for i in ann['images'] if i['file_name'] not in CORRUPT_FILES]
        print(f'  [{split}] Loaded {len(self.file_names)} scans (4 classes)')

    def _load(self,fname):
        stem=fname.replace('.h5','')
        e1=clip_and_normalize(np.load(f'{self.npy_dir}/{stem}_echo1.npy'))
        e2=clip_and_normalize(np.load(f'{self.npy_dir}/{stem}_echo2.npy'))
        seg=np.load(f'{self.npy_dir}/{stem}_seg.npy')
        return np.stack([e1,e2],axis=0),seg_to_label_4class(seg)

    def __len__(self): return len(self.file_names)

    def __getitem__(self,idx):
        image,label=self._load(self.file_names[idx])
        if self.split=='train':
            if self.do_augment: image,label=augment(image,label)
            ip,lp=sample_patch_balanced(image,label,self.patch_size,self.pos_ratio)
            return {'image':torch.from_numpy(ip).float(),'label':torch.from_numpy(lp).long()}
        return {'image':torch.from_numpy(image).float(),'label':torch.from_numpy(label).long()}


class DiceFocalLoss(nn.Module):
    def __init__(self,smooth=1e-5): super().__init__(); self.smooth=smooth
    def forward(self,pred,target):
        nc=pred.shape[1]; ps=torch.softmax(pred,dim=1)
        to=F.one_hot(target.long(),nc).permute(0,4,1,2,3).float()
        dl=sum(1-(2*(ps[:,c].reshape(-1)*to[:,c].reshape(-1)).sum()+self.smooth)/
               (ps[:,c].reshape(-1).sum()+to[:,c].reshape(-1).sum()+self.smooth)
               for c in range(1,nc))/(nc-1)
        w=torch.tensor([0.1]+[1.0]*(nc-1),device=pred.device,dtype=pred.dtype)
        return dl+F.cross_entropy(pred,target.long(),weight=w/w.sum())

SPACING=(1.0,1.0,1.0)

def compute_dsc(po,lo,nc,smooth=1e-5):
    d=np.zeros(nc,dtype=np.float32)
    for c in range(1,nc+1):
        p=po[c].float();t=lo[c].float();i=(p*t).sum().item();dn=p.sum().item()+t.sum().item()
        d[c-1]=1.0 if dn==0 else (2.0*i+smooth)/(dn+smooth)
    return d

def compute_vs(po,lo,nc):
    v=np.zeros(nc,dtype=np.float32)
    for c in range(1,nc+1):
        vp=po[c].float().sum().item();vt=lo[c].float().sum().item();dn=vp+vt
        v[c-1]=1.0 if dn==0 else 1.0-abs(vp-vt)/dn
    return v

def compute_hd95_bbox(po,lo,nc,spacing=SPACING):
    h=np.zeros(nc,dtype=np.float32)
    for c in range(1,nc+1):
        p=po[c].numpy().astype(bool);t=lo[c].numpy().astype(bool)
        if not p.any() or not t.any():
            H,W,D=p.shape; h[c-1]=math.sqrt((H*spacing[0])**2+(W*spacing[1])**2+(D*spacing[2])**2); continue
        coords=np.argwhere(p|t); h0,w0,d0=coords.min(axis=0); h1,w1,d1=coords.max(axis=0); m=5
        pc=p[max(0,h0-m):min(p.shape[0],h1+m+1),max(0,w0-m):min(p.shape[1],w1+m+1),max(0,d0-m):min(p.shape[2],d1+m+1)]
        tc=t[max(0,h0-m):min(t.shape[0],h1+m+1),max(0,w0-m):min(t.shape[1],w1+m+1),max(0,d0-m):min(t.shape[2],d1+m+1)]
        dp=distance_transform_edt(~pc,sampling=spacing); dt=distance_transform_edt(~tc,sampling=spacing)
        sp=pc&(distance_transform_edt(pc,sampling=spacing)==1); st=tc&(distance_transform_edt(tc,sampling=spacing)==1)
        if not sp.any(): sp=pc
        if not st.any(): st=tc
        h[c-1]=np.percentile(np.concatenate([dt[sp],dp[st]]),95)
    return h

def sliding_window_inference(model,image,patch_size,num_classes,overlap=0.5,device=None):
    B,C,H,W,D=image.shape;pH,pW,pD=patch_size;oc=num_classes+1
    ph=max(0,pH-H);pw=max(0,pW-W);pd=max(0,pD-D)
    if ph>0 or pw>0 or pd>0:
        image=F.pad(image,(pd//2,pd-pd//2,pw//2,pw-pw//2,ph//2,ph-ph//2));_,_,H,W,D=image.shape
    out=torch.zeros(B,oc,H,W,D,device=device);cnt=torch.zeros(B,1,H,W,D,device=device)
    def gs(s,p,st):
        r=list(range(0,s-p+1,st))
        if not r or r[-1]+p<s: r.append(max(0,s-p))
        return r
    sh=max(1,int(pH*(1-overlap)));sw=max(1,int(pW*(1-overlap)));sd=max(1,int(pD*(1-overlap)))
    with torch.no_grad():
        for h0 in gs(H,pH,sh):
            for w0 in gs(W,pW,sw):
                for d0 in gs(D,pD,sd):
                    pred=model(image[:,:,h0:h0+pH,w0:w0+pW,d0:d0+pD])
                    out[:,:,h0:h0+pH,w0:w0+pW,d0:d0+pD]+=pred
                    cnt[:,:,h0:h0+pH,w0:w0+pW,d0:d0+pD]+=1
    res=out/cnt.clamp(min=1)
    if ph>0 or pw>0 or pd>0:
        res=res[:,:,ph//2:ph//2+(H-ph),pw//2:pw//2+(W-pw),pd//2:pd//2+(D-pd)]
    return res

def save_checkpoint(epoch,model,optimizer,scheduler,history,best_dice,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'history':history,'best_dice':best_dice},path)

def load_checkpoint(path,model,optimizer,scheduler,device):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'],ckpt['history'],ckpt['best_dice']

def train_one_epoch(model,loader,optimizer,loss_fn,device):
    model.train(); el=0.0; step=0
    pbar=tqdm(loader,desc='  Training UNet3D',leave=False)
    for batch in pbar:
        images=batch['image'].to(device); labels=batch['label'].to(device)
        optimizer.zero_grad()
        logits=model(images); loss=loss_fn(logits,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
        el+=loss.item(); step+=1; pbar.set_postfix({'loss':f'{el/step:.4f}'})
        del images,labels,logits,loss; torch.cuda.empty_cache()
    return el/step

def validate(model,loader,device):
    model.eval(); dl,hl,vl=[],[],[]
    with torch.no_grad():
        for batch in tqdm(loader,desc='  Validating UNet3D',leave=False):
            images=batch['image'].to(device); labels=batch['label'].to(device)
            torch.cuda.empty_cache()
            out=sliding_window_inference(model,images,PATCH_SIZE,NUM_CLASSES,OVERLAP,device)
            pi=out.argmax(dim=1).cpu(); lc=labels.cpu()
            del out,images,labels; torch.cuda.empty_cache()
            for b in range(pi.shape[0]):
                po=F.one_hot(pi[b],NUM_CLASSES+1).permute(3,0,1,2)
                lo=F.one_hot(lc[b],NUM_CLASSES+1).permute(3,0,1,2)
                dl.append(compute_dsc(po,lo,NUM_CLASSES)); vl.append(compute_vs(po,lo,NUM_CLASSES))
                hl.append(compute_hd95_bbox(po,lo,NUM_CLASSES)); del po,lo
            del pi,lc
    return np.mean(dl,axis=0),np.mean(hl,axis=0),np.mean(vl,axis=0)

if __name__=='__main__':
    os.makedirs(LOG_DIR,exist_ok=True); os.makedirs(CKPT_DIR,exist_ok=True); os.makedirs(RESULTS_DIR,exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {DEVICE}'); print(f'GPU    : {torch.cuda.get_device_name(0)}')
    torch.cuda.set_per_process_memory_fraction(0.95)

    print('\nLoading datasets (4 classes)...')
    train_ds=SKMTEADataset('train',NPY_DIR,ANNOT_DIR,PATCH_SIZE,pos_ratio=0.8,do_augment=True)
    val_ds  =SKMTEADataset('val',  NPY_DIR,ANNOT_DIR,PATCH_SIZE,do_augment=False)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=False)
    val_loader  =DataLoader(val_ds,  batch_size=1,shuffle=False,num_workers=NUM_WORKERS,pin_memory=False)

    print('\nInitializing UNet3D (4 classes)...')
    model=UNet3D(IN_CHANNELS,NUM_CLASSES,CHANNELS,DROPOUT).to(DEVICE)
    total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters  : {total:,}')
    print(f'Num classes : {NUM_CLASSES} (Patellar, Femoral, Tibial, Meniscus)')
    print(f'Channels    : {CHANNELS}')
    print(f'Patch size  : {PATCH_SIZE}')
    print(f'Batch size  : {BATCH_SIZE}')
    print(f'LR          : {LR}')

    loss_fn  =DiceFocalLoss()
    optimizer=optim.Adam(model.parameters(),lr=LR,betas=BETAS)
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer,gamma=LR_GAMMA)

    history={'train_loss':[],'val_dice':[],'val_dice_per':[],'val_hd95':[],'val_hd95_per':[],
             'val_vs':[],'val_vs_per':[],'val_epochs':[]}
    best_dice=0.0; start_epoch=1; early_stop_count=0

    if RESUME and os.path.exists(CKPT_PATH):
        start_epoch,history,best_dice=load_checkpoint(CKPT_PATH,model,optimizer,scheduler,DEVICE)
        start_epoch+=1; print(f'Resumed epoch {start_epoch-1}, best DSC={best_dice:.4f}')
    else:
        print('Starting from scratch')

    print(f'\nStarting training — {MAX_EPOCHS} epochs'); print('='*65)

    for epoch in range(start_epoch,MAX_EPOCHS+1):
        tl=train_one_epoch(model,train_loader,optimizer,loss_fn,DEVICE)
        scheduler.step(); history['train_loss'].append(tl)

        if epoch%VAL_INTERVAL==0:
            d,h,vs=validate(model,val_loader,DEVICE)
            history['val_dice'].append(float(d.mean())); history['val_dice_per'].append(d.tolist())
            history['val_hd95'].append(float(h.mean())); history['val_hd95_per'].append(h.tolist())
            history['val_vs'].append(float(vs.mean()));  history['val_vs_per'].append(vs.tolist())
            history['val_epochs'].append(epoch)
            print(f'\n{"─"*65}\nEpoch {epoch:3d}/{MAX_EPOCHS}')
            print(f'  Train Loss : {tl:.4f}')
            print(f'  Val   DSC  : {d.mean():.4f}  HD95: {h.mean():.2f}mm  VS: {vs.mean():.4f}')
            for i,n in enumerate(LABEL_NAMES):
                print(f'    {n:<12}: DSC={d[i]:.4f}  HD95={h[i]:.2f}mm  VS={vs[i]:.4f}')
            if d.mean()>best_dice:
                best_dice=d.mean(); early_stop_count=0
                torch.save(model.state_dict(),BEST_PATH)
                print(f'  * Best model saved! (DSC={best_dice:.4f})')
            else:
                early_stop_count+=VAL_INTERVAL
                print(f'  No improvement. Early stop: {early_stop_count}/{EARLY_STOP_PAT}')
                if early_stop_count>=EARLY_STOP_PAT:
                    print(f'\n  Early stopping at epoch {epoch}! Best DSC: {best_dice:.4f}'); break
        else:
            print(f'\n{"─"*65}\nEpoch {epoch:3d}/{MAX_EPOCHS}  Train Loss : {tl:.4f}')

        if epoch%SAVE_INTERVAL==0:
            save_checkpoint(epoch,model,optimizer,scheduler,history,best_dice,CKPT_PATH)
            with open(os.path.join(RESULTS_DIR,'unet3d_4class_history.json'),'w') as f:
                json.dump(history,f,indent=2)
            print(f'  [Checkpoint] Saved at epoch {epoch}')

    print(f'\n{"="*65}\nTraining complete! Best Val DSC: {best_dice:.4f}')
    with open(os.path.join(RESULTS_DIR,'unet3d_4class_history.json'),'w') as f:
        json.dump(history,f,indent=2)
