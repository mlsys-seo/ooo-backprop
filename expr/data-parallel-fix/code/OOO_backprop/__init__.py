import time

from .arguments import get_args

args = get_args()

bcolors = {
  'P': '\033[95m',
  'B': '\033[94m',
  'C': '\033[96m',
  'G': '\033[92m',
  'Y': '\033[93m',
  'R': '\033[91m',
  'ENDC': '\033[0m',
  'BOLD': '\033[1m',
  'UNDERLINE': '\033[4m',
}

def print_log(message, color='G'):
    print(f"{bcolors[color]}{message}{bcolors['ENDC']}")

ITER_SUM = 0
ITER_COUNT = 0
IS_FIRST = 1

class print_timestep(object):
  def __init__(self, message, average=True, color='Y'):
    self.message = message
    self.color = bcolors[color]

    self.average = average

  def __enter__(self):
    self.start_time = time.time()

  def __exit__(self, type, value, traceback):
    self.end_time = time.time()
    iter_time = self.end_time - self.start_time
    
    global ITER_SUM
    global ITER_COUNT
    global IS_FIRST

    if (ITER_COUNT < 10) or self.average == False:
      print(f"{self.color}{self.message}: {round(iter_time, 4)} sec{bcolors['ENDC']}")

    else:
      ITER_SUM = ITER_SUM + iter_time
      print(f"{self.color}{self.message}: {round(iter_time, 4)} sec | avg: {ITER_SUM/ITER_COUNT} / iter {bcolors['ENDC']}")

    ITER_COUNT = ITER_COUNT + 1

def print_graph(graph):
  print("\n\n======= print ops in dp_schedule ========")
  conv_ops_list = []
  async_count = 0
  for op in graph.get_operations():
      if "CONV_LAYER_" in op.name and "PushPull" not in op.name and "Conv2D" in op.name and "ShapeN" not in op.name:
        conv_ops_list.append(op)
        if args.async_op in op.name:
            async_count += 1
  
  print_log(f"num of CONV: {len(conv_ops_list)}, num of Async_CONV: {async_count}\n\n\n\n\n", "P")

  for op in conv_ops_list:
      print_log(f"OP NAME: {op.name}", "B")
      print_log(f"INPUTS: {[ target.name for target in op.inputs ]}", "Y")
      print_log(f"OUTPUTS: {[ target.name for target in op.outputs ]}", "G")
      print_log(f"CONTROL_INPUTS: {[ target.name for target in op.control_inputs ]}", "C")
      print(" ")
  print("=========================================")