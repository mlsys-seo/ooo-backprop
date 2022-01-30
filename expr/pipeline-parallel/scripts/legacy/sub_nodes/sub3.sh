export grpc_master=localhost
grpc_worker0=localhost
grpc_worker1=localhost
grpc_worker2=localhost

task_index=3

# worker_hosts=$grpc_master:2232,$grpc_worker0:2232
worker_hosts=$grpc_master:2232,$grpc_worker0:2232,$grpc_worker1:2232,$grpc_worker2:2232

python sub_node.py \
    --task_index=${task_index} \
    --worker_hosts=$worker_hosts
