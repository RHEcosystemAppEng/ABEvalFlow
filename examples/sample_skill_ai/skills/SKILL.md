# Hello World Skill

A trivial file-creation skill used as a smoke test for the ABEvalFlow pipeline.

## What the agent must do

Given an instruction to create a file with specific content, the agent should
produce the correct file at the correct path with the exact expected content.

## Evaluation criteria

- File exists at the specified path
- File content matches the expected string exactly (including punctuation)
