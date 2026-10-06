"""
Locust load-testing script for Infra Watchdog.

Ramps traffic from light to heavy load against /api/simulate so cpu_usage,
active_connections, and lag_ms sweep across their full realistic range --
giving the regression model examples of both healthy and degraded states,
not just one operating point.

Interactive (web UI at http://localhost:8089):
    locust -f locustfile.py --host http://<EC2_PUBLIC_IP>:8000

Headless, using the staged shape below:
    locust -f locustfile.py --host http://<EC2_PUBLIC_IP>:8000 --headless \
        --csv locust_report
"""

import random

from locust import HttpUser, LoadTestShape, task, between


class BackendUser(HttpUser):
    wait_time = between(0.05, 0.5)

    @task
    def hit_simulate(self):
        intensity = random.randint(1, 10)
        self.client.get(f"/api/simulate?intensity={intensity}", name="/api/simulate")


class RampUpShape(LoadTestShape):
    """
    Staged ramp: 0 -> 300 concurrent users over 15 minutes, then cool down.
    Only takes effect in headless mode (or if you press "start" with no
    manual user count set in the web UI).
    """

    stages = [
        {"duration": 120, "users": 20, "spawn_rate": 2},
        {"duration": 240, "users": 60, "spawn_rate": 2},
        {"duration": 360, "users": 120, "spawn_rate": 3},
        {"duration": 540, "users": 200, "spawn_rate": 3},
        {"duration": 720, "users": 300, "spawn_rate": 5},
        {"duration": 900, "users": 0, "spawn_rate": 10},  # cool down
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
