import os
from os import listdir
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from data.data_utils import is_image_file
from model.DRSN import DRSN as Net
import scipy.io as scio  
from data.eval import PSNR, SSIM, SAM, RMSE,CC
import cv2

                    
def main():

    input_path = '/data/HSI/' + opt.datasetName + '/test/' + str(opt.upscale_factor) + '/' 
    out_path = '/data/HSI/' +  opt.datasetName + '/output/ours/' + str(opt.upscale_factor) + '/' 
    
    PSNRs,SSIMs,SAMs,RMSEs,CCs = [], [], [], [], []

    if not os.path.exists(out_path):
        os.makedirs(out_path)
                
    if opt.cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = Net('cuda',opt.upscale_factor, opt.depth, opt.bands_eve, opt.n_feats)  

    if opt.cuda:
        model = nn.DataParallel(model).cuda()   
 
    checkpoint  = torch.load(opt.model_name)
    model.load_state_dict(checkpoint["model"])  

    images_name = [x for x in listdir(input_path) if is_image_file(x)]           
   
    for index in range(len(images_name)):

        mat = scio.loadmat(input_path + images_name[index]) 
        hyperLR = mat['LR'].astype(np.float32)   # [h,w,c]

        h,w,c = hyperLR.shape
        hyperlms = cv2.resize(hyperLR,(w*opt.upscale_factor,h*opt.upscale_factor), interpolation=cv2.INTER_CUBIC)
        hyperlms = hyperlms.transpose(2,0,1)
        hyperLR = hyperLR.transpose(2,0,1)

        model.eval()
        with torch.no_grad():   	        	
            input = Variable(torch.from_numpy(hyperLR).float()).contiguous().view(1, -1, hyperLR.shape[1], hyperLR.shape[2])
            lms =  Variable(torch.from_numpy(hyperlms).float()).contiguous().view(1, -1, hyperlms.shape[1], hyperlms.shape[2])
            if opt.cuda:
                input = input.cuda()                
            output = model(input)   
            HR = mat['HR'].transpose(2,0,1).astype(np.float32)    
            SR = output.cpu().data[0].numpy().astype(np.float32)        
            SR[SR<0] = 0             
            SR[SR>1.] = 1.

        psnr = PSNR(SR, HR)
        sam = SAM(SR, HR)
        ssim = SSIM(SR, HR)
        rmse = RMSE(SR,HR)
        cc = CC(SR,HR)

        
        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)
        RMSEs.append(rmse)
        CCs.append(cc)
        
        SR = SR.transpose(1,2,0)   
        HR = HR.transpose(1,2,0)  
	                    
        scio.savemat(out_path + images_name[index], {'HR': HR, 'SR':SR})  
        print("=====The {}-th picture===PSNR:{:.4f}===SSIM:{:.4f}===SAM:{:.4f}===RMSE:{:.4f}===CC:{:.4f}===Name:{}".format(index+1,
          psnr, ssim, sam,rmse,cc,images_name[index])) 
    print("=====averPSNR:{:.4f}===averSSIM:{:.4f}===averSAM:{:.4f}===averRMSE:{:.4f}===averCC:{:.4f}".format(np.mean(PSNRs), 
    np.mean(SSIMs), np.mean(SAMs),np.mean(RMSEs),np.mean(CCs))) 
    
if __name__ == "__main__":
    main()
