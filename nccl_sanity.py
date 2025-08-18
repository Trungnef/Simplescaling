import torch, torch.distributed as dist, torch.multiprocessing as mp

def run(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    x = torch.ones(1, device=rank)
    dist.all_reduce(x)
    print(f"rank{rank} ok:", x.item())
    dist.destroy_process_group()

if __name__ == "__main__":
    ws = torch.cuda.device_count()
    mp.spawn(run, args=(ws,), nprocs=ws)
