"""Evaluates temporal test on synthetic data."""
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import random
import socket
import time
import sys

root_path = os.getcwd()
sys.path.append(root_path)
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from metrics import retrieval
from args import get_args
from loader.synthetic import Synthetic
from s3dg import S3D
from tqdm import tqdm
import numpy as np
import time
from utils import AllGather
allgather = AllGather.apply


def get_loader(args, text_key="caption"):
    test_dataset = Synthetic(
        metadata_dir="/ssd/pbagad/datasets/ToT-syn-v2.0/metadata",
        video_root="/ssd/pbagad/datasets/ToT-syn-v2.0/videos",
        text_key=text_key,
        num_clip=args.num_windows_test,
        fps=args.fps,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=False, center_crop=True,
    )
    print("Sample text: ", test_dataset[0]['text'])
    
    if args.parallelize:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            shuffle=False,
            drop_last=False, 
            num_workers=args.num_thread_reader,
            sampler=test_sampler,
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_thread_reader,
        )
    return test_loader


def main(args):
    model = deploy_model(args)
    
    # small sanity check to show that the model is insensitive to word order
    text_emb1 = model.text_module(["This is a test"], raw_text=True)
    text_emb2 = model.text_module(["is a This test"], raw_text=True)
    assert torch.allclose(text_emb1, text_emb2, atol=1e-5)
    from termcolor import colored
    from time import sleep
    print()
    print(colored("***** WARNING: This model is insensitive to word order *****", "red"))
    print()
    sleep(30)

    # caption similarity
    caption_test_loader = get_loader(args, text_key="caption")
    caption_all_video_embd, caption_all_text_embd = test(caption_test_loader, model, args)

    if args.gpu == 0:
        # compute similarity
        caption_v2t_sim = np.dot(caption_all_video_embd, caption_all_text_embd.T)
        # take the diagonal
        caption_v2t_sim = np.diag(caption_v2t_sim)

    # distractor similarity
    distractor_test_loader = get_loader(args, text_key="distractor")
    distractor_all_video_embd, distractor_all_text_embd = test(distractor_test_loader, model, args)

    if args.gpu == 0:
        # compute similarity
        distractor_v2t_sim = np.dot(distractor_all_video_embd, distractor_all_text_embd.T)
        # take the diagonal
        distractor_v2t_sim = np.diag(distractor_v2t_sim)

        print(caption_v2t_sim)
        print(distractor_v2t_sim)
        accuracy = np.mean(caption_v2t_sim > distractor_v2t_sim)
        print("Accuracy: ", accuracy)

    #     t2v = retrieval(np.dot(all_text_embd, all_video_embd.T))
    #     v2t = retrieval(np.dot(all_video_embd, all_text_embd.T))
    #     print('MSRVTT')
    #     print(f"R@1: {t2v['R1']:.2f} - R@5: {t2v['R5']:.2f} - R@10: {t2v['R10']:.2f} - Median R: {t2v['MR']}")
    #     print(f"R@1: {v2t['R1']:.2f} - R@5: {v2t['R5']:.2f} - R@10: {v2t['R10']:.2f} - Median R: {v2t['MR']}")
    #     with open('result.txt', 'a') as f:
    #         f.write('MSRVTT\n')
    #         f.write(f"R@1: {t2v['R1']:.2f} - R@5: {t2v['R5']:.2f} - R@10: {t2v['R10']:.2f} - Median R: {t2v['MR']}\n")
    #         f.write(f"R@1: {v2t['R1']:.2f} - R@5: {v2t['R5']:.2f} - R@10: {v2t['R10']:.2f} - Median R: {v2t['MR']}\n")

def test(test_loader, model, args):
    all_text_embd = []
    all_video_embd = []
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(test_loader)):
            text = data['text'].cuda()
            video = data['video'].float().cuda()
            video = video / 255.0
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
            video_embd, text_embd = model(video, text)
            video_embd = video_embd.view(text_embd.shape[0], args.num_windows_test, text_embd.shape[1])
            video_embd = video_embd.mean(dim=1)
            all_text_embd.append(text_embd)
            all_video_embd.append(video_embd)
    all_text_embd = torch.cat(all_text_embd, dim=0)
    all_video_embd = torch.cat(all_video_embd, dim=0)
    
    if args.parallelize:
        all_text_embd = allgather(all_text_embd)
        all_video_embd = allgather(all_video_embd)

    return all_video_embd.cpu().numpy(), all_text_embd.cpu().numpy()
    

def deploy_model(args):
    checkpoint_path = args.pretrain_cnn_path
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    torch.cuda.set_device(args.gpu)
    model = S3D(args.num_class, space_to_depth=False, word2vec_path=args.word2vec_path)
    model.cuda(args.gpu)
    checkpoint_module = {k[7:]:v for k,v in checkpoint.items()}
    model.load_state_dict(checkpoint_module)
    
    if args.parallelize:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    model.eval()
    print(f'Model Loaded on GPU {args.gpu}')
    return model

def main_worker(gpu, ngpus_per_node, main, args):
    cudnn.benchmark = True
    args.gpu = gpu
    args.rank = gpu
    args.world_size = 4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    args.dist_url = f'tcp://{ip}:12345'
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=ngpus_per_node, rank=gpu)
    main(args)

def spawn_workers(main, args):
    ngpus_per_node = 4
    args.world_size = 4
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, main, args))

if __name__ == "__main__":
    args = get_args()
    args.fps = 30
    args.num_windows_test = 1
    
    assert args.eval_video_root != ''
    
    parallelize = False
    args.parallelize = parallelize
    
    if parallelize:
        spawn_workers(main, args)
    else:
        main(args)