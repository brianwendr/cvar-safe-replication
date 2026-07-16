#!/usr/bin/env bash
set -euo pipefail
kind create cluster --name cvar-safe --config "$(dirname "$0")/kind-config.yaml"
docker build -f "$(dirname "$0")/../Dockerfile" -t cvar-safe-demo:local "$(dirname "$0")/../.."
kind load docker-image cvar-safe-demo:local --name cvar-safe
kubectl cluster-info --context kind-cvar-safe
