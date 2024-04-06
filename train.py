import torch
import clip_local
import data_helper
from PIL import Image
import torchvision
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
from config import get_cfg
# import loralib 
import random
# from cls_name import classnames_domainnet,classnames_officehome
import torch.utils.data.dataloader
import numpy as np
from clip_local import clip
import loralib
import os
import matplotlib.pyplot as plt



def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def test(target_loader,model,text_features_norm):
    total_correct = 0
    total = 0
    for i,(x,y) in tqdm(enumerate(target_loader),total=len(target_loader)):
        x = x.to(args.device)
        y = y.to(args.device)
        image_features = model.encode_image(x)
        image_features_norm = image_features/image_features.norm(dim=-1,keepdim=True)
        logits = image_features_norm @ text_features_norm.T
        pred = torch.argmax(logits,dim=-1,keepdim=False)
        correct = torch.sum(pred==y)
        total_correct += correct
        total += x.shape[0]
    test_acc = total_correct/total
    return test_acc



def main(args):
    cfg = get_cfg(args.ds)
    cfg.LR = args.lr
    result_file = open(os.getcwd()+'/target:{}-ds:{}-lr:{}-layers:{}-R:{}.txt'.format(args.target,args.ds,cfg.LR,cfg.LORA_LAYERS,cfg.LORA_R),'a')
    model, preprocess = clip_local.load("ViT-B/16", device=args.device,lora_r=cfg.LORA_R,lora_layers=cfg.LORA_LAYERS)
    model.visual.transformer.set_lora_params()
    loralib.mark_only_lora_as_trainable(model)
    for n,p in model.named_parameters():
        if p.requires_grad == True:
            print(n)
    text_feature_norm = torch.load('/root/lora_clip/cls_feature_officehome_ViT-B-16.pt',map_location=args.device)

    if args.ds == 'domainnet':
        source_loader, target_loader = data_helper.get_loader_domainnet(transform=preprocess,bs=cfg.BS,target=args.target)
        # classnames = classnames_domainnet
    else:
        source_loader, target_loader = data_helper.get_loader_officehome(transform=preprocess,bs=cfg.BS,target=args.target)
        # classnames = classnames_officehome
        
    print('using dataset: {}'.format(args.ds))  
    print('using dataset: {}'.format(args.ds),file=result_file)  
    print(cfg)
    print(args)
    print(cfg,file=result_file)
    print(args,file=result_file)
    
    epochh = []
    acc = []
    
    optim = torch.optim.Adam(model.visual.transformer.parameters(),lr=cfg.LR,eps=1e-8)
    optimw = torch.optim.AdamW(model.visual.transformer.parameters(),lr=cfg.LR,eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()  
    
    model.float()
    text_feature_norm = text_feature_norm.to(dtype=torch.float32)
    
    best = [0,0]
    print(test(target_loader,model,text_feature_norm))
    acc.append(test(target_loader,model,text_feature_norm).cpu().numpy())
    
    for epoch in range(cfg.EPOCHS):  
        model.train()
        total_loss = 0
        total = 0
        for i,(x,y) in tqdm(enumerate(source_loader),total=len(source_loader)):
            x,y = x.to(args.device),y.to(args.device)
            image_features = model.encode_image(x)
            image_features_norm = image_features/image_features.norm(dim=-1,keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features_norm @ text_feature_norm.T
            loss = criterion(logits,y)
            optimw.zero_grad()
            loss.backward()
            optimw.step()
            total += 1
            total_loss += loss.item()

        model.eval()
        current_acc = test(target_loader,model,text_feature_norm)
        acc.append(current_acc.cpu().numpy())
        if current_acc > best[1]:
            best[0] = epoch
            best[1] = current_acc
            # torch.save(model.state_dict(),os.getcwd()+'/target:{}-ds:{}-lr:{}-layers:{}-R:{}.pt'.format(args.target,args.ds,cfg.LR,cfg.LORA_LAYERS,cfg.LORA_R))
        print('epoch:',epoch,'loss:',total_loss/total,'current_acc:',current_acc,'current_best:',best[1],'best_epoch:',best[0],file=result_file)
        print('epoch:',epoch,'loss:',total_loss/total,'current_acc:',current_acc,'current_best:',best[1],'best_epoch:',best[0])
    for i in range(cfg.EPOCHS + 1):
        epochh.append(i)
    plt.figure()         
    plt.xlabel('epoch')
    plt.plot(epochh,acc)
    plt.legend()
    plt.grid()
    plt.savefig(os.getcwd()+'/target:{}-ds:{}-lr:{}-layers:{}-R:{}.png'.format(args.target,args.ds,cfg.LR,cfg.LORA_LAYERS,cfg.LORA_R))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=int)
    parser.add_argument('--device',type=str)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--ds',type=str)
    parser.add_argument('--lr',type=float)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
 
