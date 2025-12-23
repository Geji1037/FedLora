import ray

ray.init(address="auto")
print("OK connected to cluster")

print("cluster resources:", ray.cluster_resources())
print("nodes:")
for n in ray.nodes():
    print(n["NodeID"], n["NodeManagerAddress"], n["Alive"])

