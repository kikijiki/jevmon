#!/bin/sh
# Harmless lab payload for the download-and-run scenario shape: sleep and print. Served only by
# the lab's own nginx on its internal network; executing it does nothing else.
sleep 3
echo lab-payload-done
