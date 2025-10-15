import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
 
from option import opt
from model.DRSN import DRSN as Net
from model.loss_utils import HLoss
from data.data_utils import TrainsetFromFolder,ValsetFromFolder
from data.eval import PSNR,SAM,SSIM
from torch.optim.lr_scheduler import MultiStepLR

import torch.distributed as dist


def main():
    if opt.show:
        if not os.path.exists("logs/"):
            os.makedirs("logs/")
        global writer
        writer = SummaryWriter(log_dir="logs/") 

    if opt.cuda:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    
    # TODO: Set training dataset and validation dataset
    train_set = TrainsetFromFolder('/data2/HSI/'+ opt.datasetName + '/train/' + str(opt.upscale_factor) + '/')
    val_set = ValsetFromFolder('/data2/HSI/' + opt.datasetName + '/val/' + str(opt.upscale_factor)+ '/')
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=opt.threads,batch_size=opt.batchSize, sampler=train_sampler,shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=opt.threads,batch_size=1, sampler=val_sampler,shuffle=False)

    model = Net(device,opt.upscale_factor,opt.n_bands,opt.depth,opt.bands_eve,opt.n_feats)     
    model = model.to(device)
    L1Loss = HLoss(0.5,0.1) 
    #L1Loss = nn.L1Loss()

    if opt.cuda:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        L1Loss = L1Loss.to(device)
    else:
        model = model.cpu()
    print('# parameters:', sum(param.numel() for param in model.parameters())) 

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.02)
    #optimizer=optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.02, eps=1e-08)

    if opt.resume:
        if os.path.isfile(opt.resume):   
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)               
            opt.start_epoch = checkpoint['epoch'] + 1         
            model.load_state_dict(checkpoint['model'])         
            optimizer.load_state_dict(checkpoint['optimizer']) 
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    scheduler = MultiStepLR(optimizer, milestones=[120,180,240], gamma=0.5, last_epoch=-1) 
   
    if rank == 0:
        print("            =======  Training  ======= \n")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        start_time = time.time()
        scheduler.step()  
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(train_loader, optimizer, model, L1Loss, epoch,device) 

        if(epoch%2==0):
            val(val_loader, model, epoch)                           
        if(epoch%10==0):
            save_checkpoint(epoch, model, optimizer)  
        print("*** This epoch training was completed within %.3fs***" % (time.time() - start_time))

    print("Done")           


def train(train_loader, optimizer, model, L1Loss, epoch,device):
    model.train()  
    for iteration, batch in enumerate(train_loader, 1):  
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)   

        if opt.cuda:
            input = input.to(device)
            label = label.to(device)
        SR = model(input)          

        Loss = L1Loss(SR, label)   
        
        optimizer.zero_grad()      
        Loss.backward()             
        
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        
        optimizer.step()            

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), Loss.data))

        if opt.show:
            niter = epoch * len(train_loader) + iteration
            if niter % 100 == 0:
                writer.add_scalar('Train/Loss', Loss.data, niter)


def val(val_loader, model, epoch):
	            
    model.eval()  
    val_ssim = 0
    val_sam = 0

    for iteration, batch in enumerate(val_loader, 1):
        with torch.no_grad():    
            input, HR = Variable(batch[0]),  Variable(batch[1])

            if opt.cuda:
                input = input.cuda()
                HR = HR.cuda()
            
            SR = model(input) 

            val_psnr += PSNR(SR.cpu().data[0].numpy(), HR.cpu().data[0].numpy())
            val_ssim += SSIM(SR.cpu().data[0].numpy(), HR.cpu().data[0].numpy())
            val_sam += SAM(SR.cpu().data[0].numpy(), HR.cpu().data[0].numpy())
    val_psnr = val_psnr / len(val_loader) 
    val_ssim = val_ssim / len(val_loader)
    val_sam = val_sam / len(val_loader)
    print("PSNR = {:.3f}".format(val_psnr),",  SSIM = {:.3f}".format(val_ssim),",  SAM = {:.3f}".format(val_sam))  
    
    if opt.show:
        writer.add_scalar('Val/PSNR',val_psnr, epoch)    


def save_checkpoint(epoch, model, optimizer):
    model_out_path = "checkpoint/" + "{}_{}_epoch_{}.pth".format(opt.datasetName,opt.upscale_factor,epoch)
    state = {"epoch": epoch , "model": model.state_dict(), "optimizer":optimizer.state_dict()} 
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")     	
    torch.save(state, model_out_path)  

if __name__ == "__main__":
    main()

