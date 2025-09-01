# 这是主节点上运行的查找其他客户端节点上是否配置好沐曦硬件环境

import os, ray
ray.init(address="auto")

COMMON_ENV = {
    "MACA_PATH": "/opt/maca",
    "LD_LIBRARY_PATH": "/opt/maca/lib:/opt/maca/lib64:/usr/local/lib:" + os.environ.get("LD_LIBRARY_PATH",""),
    "PATH": "/opt/maca/bin:" + os.environ.get("PATH","")
}

@ray.remote(num_gpus=1, resources={"client_node_1": 1}, runtime_env={"env_vars": COMMON_ENV})
def probe():
    import os, socket, torch
    info = {
        "host": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "MACA_PATH": os.environ.get("MACA_PATH"),
        "LD_LIBRARY_PATH_has_maca": "/opt/maca" in (os.environ.get("LD_LIBRARY_PATH") or ""),
        "has_/dev/mxcd": os.path.exists("/dev/mxcd"),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
    return info

@ray.remote(num_gpus=1, resources={"client_node_2": 1}, runtime_env={"env_vars": COMMON_ENV})
def probe2():
    import os, socket, torch
    return {
        "host": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "MACA_PATH": os.environ.get("MACA_PATH"),
        "LD_LIBRARY_PATH_has_maca": "/opt/maca" in (os.environ.get("LD_LIBRARY_PATH") or ""),
        "has_/dev/mxcd": os.path.exists("/dev/mxcd"),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }

print("client_node_1 ->", ray.get(probe.remote()))
print("client_node_2 ->", ray.get(probe2.remote()))
