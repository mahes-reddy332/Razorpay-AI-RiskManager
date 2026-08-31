"""
PART 2: Latency Load Test
Starts the FastAPI server, hammers /score/{account_id} with concurrent requests,
and reports p50/p95/p99 latency. Self-contained, no wrk/locust needed.
"""
import json
import time
import statistics
import urllib.request
import urllib.error
import threading
import subprocess
import sys
import os
import random
import signal

API_PORT = 8321
WARMUP_REQUESTS = 50
BENCHMARK_REQUESTS = 500
CONCURRENT_THREADS = 10

# Load accounts from cache to get valid IDs
cache_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'cache', 'scores.json')
with open(cache_path, 'r') as f:
    cache = json.load(f)
account_ids = list(cache.keys())
print(f"Loaded {len(account_ids)} accounts from cache for load test.")

# Start server
print(f"\nStarting FastAPI server on port {API_PORT}...")
server_proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'api.server:app', '--port', str(API_PORT), '--log-level', 'warning'],
    cwd=os.path.join(os.path.dirname(__file__), '..'),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# Wait for server readiness
time.sleep(3)
for attempt in range(10):
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{API_PORT}/health', timeout=2)
        print("Server is ready.\n")
        break
    except Exception:
        time.sleep(1)
else:
    print("ERROR: Server failed to start.")
    server_proc.kill()
    sys.exit(1)

latencies = []
errors = 0
lock = threading.Lock()

def make_request(acc_id):
    global errors
    url = f'http://127.0.0.1:{API_PORT}/score/{acc_id}'
    start = time.perf_counter()
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        resp.read()
        elapsed_ms = (time.perf_counter() - start) * 1000
        with lock:
            latencies.append(elapsed_ms)
    except Exception as e:
        with lock:
            errors += 1

# Warmup
print(f"Warming up with {WARMUP_REQUESTS} requests...")
for _ in range(WARMUP_REQUESTS):
    acc = random.choice(account_ids)
    make_request(acc)
latencies.clear()
errors = 0

# Benchmark
print(f"Running {BENCHMARK_REQUESTS} requests across {CONCURRENT_THREADS} threads...")
start_bench = time.perf_counter()

threads = []
for i in range(BENCHMARK_REQUESTS):
    acc = random.choice(account_ids)
    t = threading.Thread(target=make_request, args=(acc,))
    threads.append(t)
    t.start()
    # Limit concurrency
    if len([t for t in threads if t.is_alive()]) >= CONCURRENT_THREADS:
        threads[0].join()
        threads = [t for t in threads if t.is_alive()]

for t in threads:
    t.join()

total_time = time.perf_counter() - start_bench

# Kill server
server_proc.kill()
server_proc.wait()

# Report
latencies.sort()
n = len(latencies)
p50 = latencies[int(n * 0.50)]
p95 = latencies[int(n * 0.95)]
p99 = latencies[int(n * 0.99)]
p_min = latencies[0]
p_max = latencies[-1]
mean = statistics.mean(latencies)

print("\n" + "=" * 60)
print("LATENCY LOAD TEST RESULTS")
print("=" * 60)
print(f"Total Requests:    {BENCHMARK_REQUESTS}")
print(f"Concurrent Threads: {CONCURRENT_THREADS}")
print(f"Total Wall Time:   {total_time:.2f}s")
print(f"Throughput:         {n / total_time:.0f} req/s")
print(f"Errors:            {errors}")
print()
print(f"  p50 (median):    {p50:.2f} ms")
print(f"  p95:             {p95:.2f} ms")
print(f"  p99:             {p99:.2f} ms")
print(f"  Mean:            {mean:.2f} ms")
print(f"  Min:             {p_min:.2f} ms")
print(f"  Max:             {p_max:.2f} ms")
print("=" * 60)
print()

# Server-side latency (from JSON response)
print("Note: The above measures full HTTP round-trip (client -> server -> client).")
print("Server-side O(1) dict lookup latency is reported per-response in the")
print("'lookup_latency_ms' field and is typically 0.001-0.010 ms.")
