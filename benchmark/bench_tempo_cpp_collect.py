import subprocess

# Just executes the compiled C++ binary and parses/collects its benchmark output

EXEC = "./build/bench_tempo_cpp"

def benchmark():
    result = subprocess.run(EXEC, capture_output=True, text=True, cwd="./")
    output = result.stdout.strip()

    theta_results = []
    expval_results = []
    for line in output.splitlines():
        if line.startswith("[theta] "):
            # Parse elements: [theta] n=8 theta_time=0.123456
            elements = line[9:].split()
            n = int(elements[0].split("=")[1])
            theta_time = float(elements[1].split("=")[1])
            theta_results.append((n, theta_time))
        elif line.startswith("[expval] "):
            # Parse elements: [expval] n=8 expval=-0.123456 expval_time=0.123456
            elements = line[9:].split()
            n = int(elements[0].split("=")[1])
            expval = float(elements[1].split("=")[1])
            expval_time = float(elements[2].split("=")[1])
            expval_results.append((n, "manual-tn-tempo-cpp", expval, expval_time))
    
    return theta_results, expval_results
