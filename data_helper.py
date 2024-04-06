
import torch
import torchvision


def get_loader_officehome(transform,bs,target):  #ACPR
    dataset_r = torchvision.datasets.ImageFolder("/root/DataSets/OfficeHome/Real",transform=transform)
    dataloader_r = torch.utils.data.DataLoader(dataset_r,batch_size=bs,shuffle=True,num_workers=8)

    dataset_a = torchvision.datasets.ImageFolder("/root/DataSets/OfficeHome/Art",transform=transform)
    dataloader_a = torch.utils.data.DataLoader(dataset_a,batch_size=bs,shuffle=True,num_workers=8)

    dataset_p = torchvision.datasets.ImageFolder("/root/DataSets/OfficeHome/Product",transform=transform)
    dataloader_p = torch.utils.data.DataLoader(dataset_p,batch_size=bs,shuffle=True,num_workers=8)

    dataset_c = torchvision.datasets.ImageFolder("/root/DataSets/OfficeHome/Clipart",transform=transform)
    dataloader_c = torch.utils.data.DataLoader(dataset_c,batch_size=bs,shuffle=True,num_workers=8)
    dataloaders = [dataloader_a,dataloader_c,dataloader_p,dataloader_r]
    datasets = [dataset_a,dataset_c,dataset_p,dataset_r]
    target_loader = dataloaders[target]
    del datasets[target]
    source_dataset = torch.utils.data.ConcatDataset(datasets)
    source_loader = torch.utils.data.DataLoader(source_dataset,batch_size=bs,shuffle=True,num_workers=8)
    return source_loader,target_loader




def get_loader_domainnet(transform,bs,target):  #cipqrs
    dataset_c = torchvision.datasets.ImageFolder("/root/DataSets/domainnet/clipart",transform=transform)
    dataloader_c = torch.utils.data.DataLoader(dataset_c,batch_size=bs,shuffle=True,num_workers=8)

    dataset_i = torchvision.datasets.ImageFolder("/root/DataSets/domainnet/infograph",transform=transform)
    dataloader_i = torch.utils.data.DataLoader(dataset_i,batch_size=bs,shuffle=True,num_workers=8)

    dataset_p = torchvision.datasets.ImageFolder("/root/DataSets/domainnet/painting",transform=transform)
    dataloader_p = torch.utils.data.DataLoader(dataset_p,batch_size=bs,shuffle=True,num_workers=8)

    dataset_q = torchvision.datasets.ImageFolder("/root/DataSets/domainnet/quickdraw",transform=transform)
    dataloader_q = torch.utils.data.DataLoader(dataset_q,batch_size=bs,shuffle=True,num_workers=8)
    
    dataset_r = torchvision.datasets.ImageFolder("/root/DataSets/domainnet/real",transform=transform)
    dataloader_r = torch.utils.data.DataLoader(dataset_r,batch_size=bs,shuffle=True,num_workers=8)
    
    dataset_s = torchvision.datasets.ImageFolder("/root/DataSets/domainnet/sketch",transform=transform)
    dataloader_s = torch.utils.data.DataLoader(dataset_s,batch_size=bs,shuffle=True,num_workers=8)
    
    dataloaders = [dataloader_c,dataloader_i,dataloader_p,dataloader_q,dataloader_r,dataloader_s]
    datasets = [dataset_c,dataset_i,dataset_p,dataset_q,dataset_r,dataset_s]
    target_loader = dataloaders[target]
    del datasets[target]
    source_dataset = torch.utils.data.ConcatDataset(datasets)
    source_loader = torch.utils.data.DataLoader(source_dataset,batch_size=bs,shuffle=True,num_workers=8)
    return source_loader,target_loader
