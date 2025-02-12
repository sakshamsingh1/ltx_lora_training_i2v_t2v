#!/bin/bash

# Number of times to run the Python script
RUNS=100


SCRIPT="precompute.py"

# Loop to run the script 100 times
for ((i=1; i<=RUNS; i++))
do
    echo ">> Running test $i..."

    # Run the Python script and wait for it to finish
    /home/eisneim/.conda/envs/_learn/bin/python "$SCRIPT"

    # Check the exit status of the Python script
    if [ $? -ne 0 ]; then
        echo ">> Test $i failed (memory usage too high or other error)."
    else
        echo ">> Test $i completed successfully."
    fi

    # Clear cached memory (requires sudo privileges)
    echo "Clearing cached memory..."
    # sudo sync  # Write data to disk
    # sudo sysctl -w vm.drop_caches=3  # Clear pagecache, dentries, and inodes

    # Wait a moment before the next run
    sleep 4
done

echo "All tests completed."