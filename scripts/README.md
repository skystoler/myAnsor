# Help
## Tune single operator / subgraph

- Intel CPU  
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 100 --target "llvm -mcpu=core-avx2"
  ```

- NVIDIA GPU  
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 100 --target "cuda"
  ```

- ARM CPU  
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 100 --target "llvm -target=arm-linux-gnueabihf -mattr=+neon" --rpc-device-key rasp4b --rpc-host kraken --rpc-port 9191
  ```

## Tune Network
- Intel CPU  
  ```
  python3 tune_network.py --network resnet-18 --n-trials 200 --target "llvm -mcpu=core-avx2"
  ```

- NVIDIA GPU  
  ```
  python3 tune_network.py --network resnet-18 --n-trials 200 --target "cuda"
  ```

- ARM CPU  
  ```
  python3 tune_network.py --network resnet-18 --n-trials 200 --target "llvm -target=arm-linux-gnueabihf -mattr=+neon" --rpc-device-key rasp4b --rpc-host kraken --rpc-port 9191
  ```

## Run single op & subgraph evaluation
- Intel CPU
  ```
  # tune
  python3 tune_op_subgraph.py --wkl all --batch-size -1 --target "llvm -mcpu=core-avx2" --n-trials-per-shape 1000 

  # replay
  python3 tune_op_subgraph.py --wkl all --batch-size -1 --target "llvm -mcpu=core-avx2" --tune false
  ```

- NVIDIA GPU
  ```
  # tune
  python3 tune_op_subgraph.py --wkl subgraph --batch-size -1 --target "cuda" --n-trials-per-shape 1000 

  # replay
  python3 tune_op_subgraph.py --wkl subgraph --batch-size -1 --target "cuda" --tune false
  ```

## Train cost model
- Train an offline cost model and use it for search
  1.  Get the trainning data (log file).
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 1000 --log-file train.json
  ```
  This command can give us the log file `train.json`.

  2. Train the offline cost model
  ```
  python3 train_cost_model.py train.json
  ```
  This command can train a cost model `saved_model.xgb`

  3. Use the model for search
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 0 --load-model saved_model.xgb
  ```
  Using `--n-trials 0` means we do not run any measurement and rely on the cost model only.
  With the pre-trained cost model, we are expected to see a pretty good result even without tuning.

  As a santiy check, we should compare the result against the output from the command below.
  The command below does not use the cost model, so it is expected to output a bad result.
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 0
  ```

## Evaluate all networks
The results will be saved to `results.tsv`
- Intel CPU
  ```
  python3 evaluate_all_networks.py --backend intel-cpu --log-file ~/Ansor-exp/saved_logs/2020-06-30-ansor-network-cpu.json
  ```
- Nvidia GPU
  ```
  python3 evaluate_all_networks.py --backend nvidia-gpu --log-file ~/Ansor-exp/saved_logs/2020-08-20-ansor-network-gpu.json
  ```

