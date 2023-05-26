import subprocess

best_returns = float('-inf')
best_e = 0
best_l = 0

for e in range(11):
    e = e / 10.0
    for l in range(11):
        l = l / 10.0
        args = ['python', 'gridworld.py', '-a', 'q', '-k', '50', '-n', '0', '-g', 'BridgeGrid', '-e', str(e), '-l', str(l)]
        result = subprocess.run(args, stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        lines = output.strip().split('\n')
        returns = float(lines[-1].split()[-1])
        print(f'e={e}, l={l}, returns={returns}')
        if returns > best_returns:
            best_returns = returns
            best_e = e
            best_l = l

print(f'Best pair: e={best_e}, l={best_l}, returns={best_returns}')
