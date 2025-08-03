#!/bin/bash

# AMP Workflow Agent Entrypoint Script

set -e

# Default values
AGENT_TYPE=${AGENT_TYPE:-"workflow-engine"}
AGENT_PORT=${AGENT_PORT:-"8080"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo "Starting AMP Workflow Agent: $AGENT_TYPE on port $AGENT_PORT"

# Wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=1

    echo "Waiting for $service at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "$service is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: $service not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: $service at $host:$port is not available after $max_attempts attempts"
    return 1
}

# Wait for Redis if state manager
if [ "$AGENT_TYPE" = "state-manager" ]; then
    wait_for_service "redis" "6379" "Redis"
fi

# Wait for core services for dependent agents
case "$AGENT_TYPE" in
    "workflow-engine")
        wait_for_service "state-manager" "8082" "State Manager"
        wait_for_service "condition-evaluator" "8083" "Condition Evaluator"
        wait_for_service "error-handler" "8084" "Error Handler"
        wait_for_service "monitor-agent" "8085" "Monitor Agent"
        ;;
    "task-executor")
        wait_for_service "workflow-engine" "8080" "Workflow Engine"
        ;;
    "web-dashboard")
        wait_for_service "workflow-engine" "8080" "Workflow Engine"
        wait_for_service "monitor-agent" "8085" "Monitor Agent"
        ;;
esac

# Set Python path
export PYTHONPATH="/app:/app/shared-lib:$PYTHONPATH"

# Start the appropriate agent
case "$AGENT_TYPE" in
    "workflow-engine")
        echo "Starting Workflow Engine..."
        exec python -m agents.workflow_engine --port $AGENT_PORT
        ;;
    "task-executor")
        echo "Starting Task Executor..."
        exec python -m agents.task_executor --port $AGENT_PORT
        ;;
    "state-manager")
        echo "Starting State Manager..."
        exec python -m agents.state_manager --port $AGENT_PORT
        ;;
    "condition-evaluator")
        echo "Starting Condition Evaluator..."
        exec python -m agents.condition_evaluator --port $AGENT_PORT
        ;;
    "error-handler")
        echo "Starting Error Handler..."
        exec python -m agents.error_handler --port $AGENT_PORT
        ;;
    "monitor-agent")
        echo "Starting Monitor Agent..."
        exec python -m agents.monitor_agent --port $AGENT_PORT
        ;;
    *)
        echo "ERROR: Unknown agent type: $AGENT_TYPE"
        echo "Valid types: workflow-engine, task-executor, state-manager, condition-evaluator, error-handler, monitor-agent"
        exit 1
        ;;
esac