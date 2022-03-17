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

def print_timestep(message, color='G'):
    print(f"{bcolors[color]}{message}{bcolors['ENDC']}")

class print_timestep(object):
  def __init__(self, message, color='Y'):
    self.message = message
    self.color = bcolors[color]

  def __enter__(self):
    self.start_time = time.time()
    # if args.debug_print == True:
    #   print(f"{self.color}{self.message}: started{bcolors['ENDC']}\n")

  def __exit__(self, type, value, traceback):
    self.end_time = time.time()
    print(f"{self.color}{self.message}: {round(self.end_time - self.start_time, 4)} sec{bcolors['ENDC']}\n")