#############################################
#              various utils                #
#          like timer for function          #
#############################################
from typing import Any
import time



nesting_level = 0
is_start = None


class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

"""
    Usage:: calculate time consumption of a function
    >>>@timeit
       def load_fulldata():
           full_data = {}
           rootdir = join(os.getcwd(), 'dataset')
           datanames = os.listdir(rootdir)
           for dataname in datanames:
               full_data[dataname.split('.')[0]] = init.load_data(rootdir, dataname)
           return full_data
           
    >>>load_fulldata()
    
    ----Start [load_fulldata]:
    Read data from file1csv.csv
    Read data from Smart meters Residential post-trial survey data.csv
    Read data from Smart meters Residential pre-trial survey data.csv
    Read data from Smart meters SME post-trial survey data.csv
    Read data from Smart meters SME pre-trial survey data.csv
    ----End   [load_fulldata]. Time elapsed: 2.28 sec.
    
    
    :param data: data from file1csv
    :return :list of 5 tables in form of pandas
    :rtype: None
"""
def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")


def show_dataframe(df):
    if len(df) <= 30:
        print(f"content=\n"
              f"{df}")
    else:
        print(f"dataframe is too large to show the content, over {len(df)} rows")

    if len(df.dtypes) <= 100:
        print(f"types=\n"
              f"{df.dtypes}\n")
    else:
        print(f"dataframe is too wide to show the dtypes, over {len(df.dtypes)} columns")