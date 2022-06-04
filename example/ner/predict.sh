#!/bin/sh

# prediction example: first train/test the model using ./harness.py traintest

SENT="West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship."

echo $SENT | ./harness.py predtext -
