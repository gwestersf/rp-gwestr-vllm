#!/usr/bin/env python3
"""
Download model weights to all RunPod network volumes in parallel.

Creates one temporary GPU pod per network volume, SSHes in to run the
HuggingFace download, then terminates each pod when done.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/download_weights.py

Requirements:
    pip install requests
    ~/.ssh/id_ed25519 registered in your RunPod account (already done)
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

MODEL_ID = "google/gemma-4-E4B-it"
LOCAL_DIR = "/workspace/models/gemma-4-E4B-it"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")

# GPU types to try in order.
# RunPod will use the first type available in the target datacenter.
GPU_TYPES = [
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
]

# Network volumes — one per datacenter
VOLUMES = [
    {"id": "aqpg4pchsw", "dc": "US-TX-3"},
    {"id": "ihzvr1kite", "dc": "US-MO-2"},
    {"id": "kcbqenq2zi", "dc": "US-MD-1"},
    {"id": "kts5i21zyy", "dc": "US-IL-1"},
    {"id": "lyso3r86r9", "dc": "US-KS-2"},
    {"id": "q3owrkjzc1", "dc": "US-GA-2"},
    {"id": "t8z13g4vef", "dc": "US-WA-1"},
    {"id": "vv1kjkh3sx", "dc": "US-NC-1"},
]

# ── GraphQL helpers ────────────────────────────────────────────────────────────

def gql(query, variables=None):
    r = requests.post(
        API_URL,
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    if "errors" in d:
        raise RuntimeError(json.dumps(d["errors"], indent=2))
    return d["data"]


_CREATE = """
mutation ($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
  }
}
"""

_GET_POD = """
query ($podId: String!) {
  pod(input: {podId: $podId}) {
    id
    desiredStatus
    runtime {
      ports {
        ip
        isIpPublic
        privatePort
        publicPort
        type
      }
    }
  }
}
"""

_TERMINATE = """
mutation ($podId: String!) {
  podTerminate(input: {podId: $podId})
}
"""

# ── Pod lifecycle ──────────────────────────────────────────────────────────────

def create_pod(volume, gpu_type):
    data = gql(_CREATE, {
        "input": {
            "cloudType": "SECURE",
            "gpuCount": 1,
            "volumeInGb": 0,
            "containerDiskInGb": 10,
            "gpuTypeId": gpu_type,
            "name": f"dl-{volume['id']}",
            "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "networkVolumeId": volume["id"],
            "startSsh": True,
            "startJupyter": False,
            "env": [
                {"key": "HF_TOKEN", "value": HF_TOKEN},
                {"key": "HUGGING_FACE_HUB_TOKEN", "value": HF_TOKEN},
            ],
        }
    })
    return data["podFindAndDeployOnDemand"]["id"]


def wait_for_ssh(pod_id, dc, timeout=600):
    """Poll until the pod exposes a public SSH port."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = gql(_GET_POD, {"podId": pod_id})["pod"]
        ports = (pod.get("runtime") or {}).get("ports") or []
        for p in ports:
            if (
                p.get("privatePort") == 22
                and p.get("isIpPublic")
                and p.get("publicPort")
            ):
                return p["ip"], int(p["publicPort"])
        time.sleep(15)
    raise TimeoutError(f"[{dc}] pod {pod_id} SSH not ready after {timeout}s")


def ssh_run(ip, port, cmd):
    return subprocess.run(
        [
            "ssh",
            "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-p", str(port),
            f"root@{ip}",
            cmd,
        ],
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max per download
    )


def terminate_pod(pod_id):
    try:
        gql(_TERMINATE, {"podId": pod_id})
    except Exception as e:
        print(f"  warn: terminate {pod_id} failed: {e}")


# ── Per-volume worker ──────────────────────────────────────────────────────────

def handle_volume(volume):
    dc = volume["dc"]
    pod_id = None
    try:
        # Try GPU types cheapest-first until one is available in this datacenter
        last_err = None
        for gpu in GPU_TYPES:
            try:
                pod_id = create_pod(volume, gpu)
                print(f"[{dc}] pod {pod_id} created ({gpu})")
                break
            except Exception as e:
                last_err = e
                print(f"[{dc}] {gpu} not available: {e}")
        else:
            raise RuntimeError(f"No GPU types available in {dc}: {last_err}")

        ip, port = wait_for_ssh(pod_id, dc)
        print(f"[{dc}] SSH ready at {ip}:{port}")

        # Idempotent: skip if already downloaded
        download_cmd = " && ".join([
            f'if [ -d "{LOCAL_DIR}" ]; then echo "Already exists, skipping"; echo DOWNLOAD_OK; exit 0; fi',
            "pip install -q huggingface_hub",
            f"huggingface-cli download {MODEL_ID} --local-dir {LOCAL_DIR} --local-dir-use-symlinks False",
            "echo DOWNLOAD_OK",
        ])

        print(f"[{dc}] Starting download...")
        result = ssh_run(ip, port, download_cmd)

        if "DOWNLOAD_OK" not in result.stdout:
            tail_out = result.stdout[-1000:].strip()
            tail_err = result.stderr[-1000:].strip()
            if tail_out:
                print(f"[{dc}] stdout tail:\n{tail_out}")
            if tail_err:
                print(f"[{dc}] stderr tail:\n{tail_err}")
            raise RuntimeError(f"[{dc}] Download did not complete — DOWNLOAD_OK not found in output")

        print(f"[{dc}] Download complete")
        return dc, True

    except Exception as e:
        print(f"[{dc}] ERROR: {e}")
        return dc, False

    finally:
        if pod_id:
            terminate_pod(pod_id)
            print(f"[{dc}] pod {pod_id} terminated")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        sys.exit("RUNPOD_API_KEY not set")
    if not HF_TOKEN:
        sys.exit("HF_TOKEN not set — Gemma is a gated model, a token is required")
    if not os.path.exists(SSH_KEY):
        sys.exit(f"SSH key not found: {SSH_KEY}")

    print(f"Downloading {MODEL_ID} → {LOCAL_DIR}")
    print(f"Volumes: {len(VOLUMES)} | GPU preference: {' > '.join(GPU_TYPES)}")
    print()

    results = {}
    with ThreadPoolExecutor(max_workers=len(VOLUMES)) as pool:
        futures = {pool.submit(handle_volume, v): v for v in VOLUMES}
        for f in as_completed(futures):
            dc, ok = f.result()
            results[dc] = ok

    print()
    print("=" * 40)
    all_ok = True
    for dc in sorted(results):
        status = "OK" if results[dc] else "FAILED"
        print(f"  {dc}: {status}")
        if not results[dc]:
            all_ok = False
    print("=" * 40)

    if not all_ok:
        sys.exit(1)
    print("All volumes complete.")
