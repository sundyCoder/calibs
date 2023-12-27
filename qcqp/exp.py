import argparse
import numpy as np 
import glob

def main(args):
    avg_dist, avg_time = [], []

    log_files = glob.glob(args.txt_path + "log*.csv")
    for log_file in log_files:
        distance, times = [], []
        with open(log_file, "r") as file:
            for line in file.readlines():
                distance.append(float(line.split(",")[1]))
                times.append(float(line.split(",")[2]))
        avg_dist.append(np.mean(distance))
        avg_time.append(np.mean(times))
    print(np.mean(avg_dist), np.var(avg_dist), np.mean(avg_time), np.var(avg_time), np.mean(avg_dist)/np.mean(avg_time))
    # print(np.mean(distance), np.var(distance), np.mean(times), np.var(times), np.mean(distance)/np.mean(times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser( prog='experiment results')
    parser.add_argument('-f', '--txt_path', default="", type=str)           # positional argument

    args = parser.parse_args()
    main(args)
