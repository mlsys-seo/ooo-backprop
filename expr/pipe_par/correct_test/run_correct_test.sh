export TF_CPP_MIN_LOG_LEVEL=1

echo "RUN BASE"
python pipe_ooo_correct_test.py 40 0

echo "RUN OOO"
python pipe_ooo_correct_test.py 40 1
python logit_diff.py

echo "RUN BASE"
python pipe_ooo_correct_test.py 80 0

echo "RUN OOO"
python pipe_ooo_correct_test.py 80 1
python logit_diff.py
