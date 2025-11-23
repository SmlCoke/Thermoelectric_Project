import subprocess
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strength analysis.")
    parser.add_argument("N", type=int, help="Time interval in minutes")
    parser.add_argument("--plot_mode", "-p", type=str, default="plot", choices=["plot", "stem"], help="Plotting mode")
    args = parser.parse_args()

    for index in range(8):
        cmd = [sys.executable, "strengthdown.py", str(args.N), str(index + 1), "--plot", args.plot_mode]
        print("Running:", cmd)
        subprocess.run(cmd, check=True)