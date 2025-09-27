import argparse


parser = argparse.ArgumentParser(description="Hyperspectral Image Super-Resolution")     
parser.add_argument("--upscale_factor", default=8, type=int, help="upscale factor")       
parser.add_argument('--seed', type=int, default=1,  help='random seed (default: 1)')      
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")      
parser.add_argument("--nEpochs", type=int, default=300, help="maximum number of epochs")  
parser.add_argument("--show", action="store_true", help="show Tensorboard")              
parser.add_argument("--lr", type=int, default=1e-4, help="initial lerning rate")       
parser.add_argument("--dropout_rate", type=int, default=0.0, help="dropout rate")       
parser.add_argument('--local_rank', default=[1,2], type=int,
                    help='node rank for distributed training')

parser.add_argument("--cuda", action="store_true", help="Use cuda")                     
parser.add_argument("--threads", type=int, default=16, help="number of threads for dataloader")    
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")   
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number")                 
parser.add_argument("--datasetName", default="CAVE", type=str, help="data name")             

parser.add_argument('--n_bands', type=int, default=31, help='number of bands')        
parser.add_argument('--n_feats', type=int, default=64, help='number of features')     
parser.add_argument('--depth', type=int, default=16, help='number of resblock')   
parser.add_argument('--bands_eve', type=int, default=4, help='bands every group')     


parser.add_argument('--model_name', default='checkpoint/model_4_epoch_200.pth', type=str, help='super resolution model name ') 
opt = parser.parse_args() 
