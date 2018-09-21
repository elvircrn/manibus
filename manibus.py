import sys

import data
import network

if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else data.DATA_PATH
    print('Data path: ', data_path)
    network.run_network(data_path)
