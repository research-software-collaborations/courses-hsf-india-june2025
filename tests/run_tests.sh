nvidia-smi

echo "building hello"
nvcc -o hello hello.cu
echo "building vadd"
nvcc -o vadd vadd.cu

echo "running hello"
./hello
echo "running vadd"
./vadd

echo "running cudnn_test"
python train_cudnn.py

echo "running pytorch"
python pytorch_test.py

echo "running jax"
chmod +x jax_test.sh
./jax_test.sh

echo "testing nsys"
nsys profile -t nvtx,cuda -o nsys_out.txt --stats=true --force-overwrite true ./hello

echo "done"
