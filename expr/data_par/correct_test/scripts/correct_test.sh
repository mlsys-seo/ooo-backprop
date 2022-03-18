echo "RUN BASE 40"

./run_2gpu_scheduler.sh &
./run_2gpu_1.sh 0 40 &
./run_2gpu_0.sh 0 40 

sleep 3

echo "RUN OOO 40"

./run_2gpu_scheduler.sh &
./run_2gpu_1.sh 1 40 &
./run_2gpu_0.sh 1 40 

sleep 5

echo "logit diff"
python3 logit_diff.py
######

echo "RUN BASE 80"

./run_2gpu_scheduler.sh &
./run_2gpu_1.sh 0 80 &
./run_2gpu_0.sh 0 80 

sleep 3

echo "RUN OOO 80"

./run_2gpu_scheduler.sh &
./run_2gpu_1.sh 1 80 &
./run_2gpu_0.sh 1 80 

sleep 5

echo "logit diff"
python3 logit_diff.py

