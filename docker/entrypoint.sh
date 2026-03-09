#!/bin/bash
# =============================================================================
# Entrypoint for MambaVision ROS2 container
# =============================================================================
#
# Sources ROS2 environment then executes whatever command is passed.
# Using "exec" replaces the shell process with the target command so that:
#   - the target process gets PID 1 (receives SIGTERM correctly on docker stop)
#   - no zombie shell process wrapping the real process
#
# If ROS2 setup file is missing (e.g. base image without ROS), we warn but
# continue — the container can still run inference without ROS.
# =============================================================================

set -e
ROS_SETUP="/opt/ros/jazzy/setup.bash"

if [ -f "$ROS_SETUP" ]; then
    # shellcheck disable=SC1090
    source "$ROS_SETUP"
else
    echo "[entrypoint] WARNING: ROS2 setup not found at $ROS_SETUP"
fi

# Source the ROS2 workspace overlay if it has been built
WS_SETUP="/ros2_ws/install/setup.bash"
if [ -f "$WS_SETUP" ]; then
    # shellcheck disable=SC1090
    source "$WS_SETUP"
fi

exec "$@"