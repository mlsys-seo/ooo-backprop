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

IS_FIRST = 1
ITER_TIME_LIST = []

def get_iter_time_list():
  global ITER_TIME_LIST
  return ITER_TIME_LIST

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
    
    global IS_FIRST
    global ITER_TIME_LIST

    if IS_FIRST == True:
      IS_FIRST = 0
    else:
      ITER_TIME_LIST.append(round(iter_time, 4))

    print(f"{self.color}{self.message}{bcolors['ENDC']}")

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