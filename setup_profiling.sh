#!/bin/bash

# Name of the tmux session
SESSION="workload-profiling"

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Start new tmux session detached
tmux new-session -d -s $SESSION

# Rename the first window
tmux rename-window -t $SESSION:0 'profiling'

# Split window horizontally for first script
tmux split-window -v -t $SESSION:0.0

# Split the right pane vertically for monitoring
tmux split-window -h -t $SESSION:0.1

# Ensure all panes are visible before sending commands
sleep 1


# Select top-right pane and run CPU monitoring
tmux select-pane -t $SESSION:0.1
tmux send-keys -t $SESSION:0.1 "python3 monitors/plotcpu.py" C-m

# Select bottom-right pane and run GPU monitoring
tmux select-pane -t $SESSION:0.2
tmux send-keys -t $SESSION:0.2 "python3 monitors/plotgpu.py" C-m

# Select left pane and run first script (gaming workload simulation)
tmux select-pane -t $SESSION:0.0
tmux send-keys -t $SESSION:0.0 "python3 realtime/main.py" C-m

# Set the layout to main-vertical and adjust pane sizes
tmux select-layout -t $SESSION main-vertical

# Resize panes to give more space to monitoring
tmux resize-pane -t $SESSION:0.0 -x 40  # Make left pane smaller
tmux resize-pane -t $SESSION:0.1 -y 20  # Make top-right pane taller
tmux resize-pane -t $SESSION:0.2 -y 20  # Make bottom-right pane taller

# Attach to the session
tmux attach-session -t $SESSION


