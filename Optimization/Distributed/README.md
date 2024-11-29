# Distribution

## Concepts
1. world
    - one world can have multiple processes, each process means one GPU
2. local rank
    - the rank of the process in the world, like GPU0, GPU1, GPU2, ...
3. servers
    - the servers that store the model parameters

## Acclerate
1. acclerate config is under 
    ~/.cache/huggingface/accelerate/default_config.yaml on macOS
