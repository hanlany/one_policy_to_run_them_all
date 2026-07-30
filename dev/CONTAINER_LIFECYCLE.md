# Docker Development Container Lifecycle

This directory provides small, single-purpose scripts for building, creating,
starting, and entering the LavaURMA development container. The container is
designed to remain alive independently of any terminal session.

This document is both a user guide and an implementation reference. An agent
adapting this pattern to another repository should follow the contracts in
[Agent implementation contract](#agent-implementation-contract).

## Quick start

Run these commands from `dev`:

```bash
./build.sh    # Build or rebuild the image.
./spawn.sh    # Create and start the container once.
./attach.sh   # Open an interactive shell in the running container.
```

Exit an attached shell with `exit` or Ctrl-D. The container continues running.
If the container is later stopped, restart it with:

```bash
./run.sh
./attach.sh
```

## Lifecycle model

The Docker image and container are different objects:

```text
docker/Dockerfile --build.sh--> image: lavaurma
                                  |
                               spawn.sh
                                  v
                         container: lavaurma_dev
                           |              |
                     stopped state   running state
                           |              |
                         run.sh        attach.sh
                           |              |
                           +------->------+---> temporary Bash session
```

| Container state | Command | Result |
| --- | --- | --- |
| Does not exist | `./spawn.sh` | Creates it in the background and leaves it running |
| Exists and is stopped | `./run.sh` | Starts the existing container |
| Exists and is running | `./attach.sh` | Opens a disposable interactive Bash session |
| Exists and is running | `./run.sh` | Reports that it is already running; no change |

Creating and starting are intentionally separate operations. Container options
such as mounts, GPU access, networking, and environment variables are fixed at
creation time. `docker start` cannot change them.

## Script roles

### `build.sh`: build the image

- Resolves paths relative to the script, so it is independent of the caller's
  current working directory.
- Uses the repository root as the Docker build context.
- Builds `docker/Dockerfile` as image `lavaurma`.
- Does not create, start, stop, or enter a container.

Rebuilding the image does not update an existing container. Recreate the
container when it must use a newly built image.

### `spawn.sh`: create and start the persistent container

- Refuses to continue if a container named `lavaurma_dev` already exists.
- Creates the host-side Bash history file before mounting it.
- Mounts the parent workspace at `/app`, uses the repository as the working
  directory, and configures host networking, optional X11 forwarding, and
  optional NVIDIA GPU access.
- Starts the container detached with `sleep infinity` as its long-lived primary
  process.
- Does not attach the user's terminal.

Because the primary process is not the user's shell, closing a terminal or
exiting an attached shell does not stop the container.

This project currently uses `--network=host`, which grants more host access
than an isolated container normally has. Retain it in another repository only
when its workload requires it.

### `run.sh`: start an existing stopped container

- Fails with a useful message if `lavaurma_dev` has never been created.
- Is idempotent when the container is already running.
- Restores local X11 authorization when applicable.
- Warns when an SSH-forwarded `DISPLAY` differs from the value captured when
  the container was created.
- Does not create a container and does not open a shell.

An SSH X11 display is part of the container's creation-time environment. If it
changes between SSH sessions and GUI programs stop working, recreate the
container from the current login.

### `attach.sh`: open a disposable interactive shell

- Requires `lavaurma_dev` to be running.
- Runs `docker exec -it lavaurma_dev bash`.
- Replaces the host script process with the Docker CLI via shell `exec`.
- Does not attach to the container's primary process.

The last distinction is the key safety property. `docker attach` connects to a
container's primary process and can affect its lifetime or input stream.
`docker exec` creates a separate process, so exiting that Bash session leaves
the persistent container running.

## Common operations

Inspect status:

```bash
docker ps --filter name=lavaurma_dev
docker ps -a --filter name=lavaurma_dev
```

Stop the container without deleting it:

```bash
docker stop lavaurma_dev
```

Recreate it after changing image or creation-time settings:

```bash
docker stop lavaurma_dev
docker rm lavaurma_dev
./spawn.sh
```

Removing a container deletes changes made only in its writable container layer.
Files under the bind-mounted workspace and the mounted `bash_history` file are
stored on the host and survive removal.

## Agent implementation contract

When rebuilding this lifecycle for another repository, preserve these
behavioral requirements even if filenames or Docker options change.

### Required inputs to identify

1. Repository root and location of the Dockerfile.
2. Image name and globally unique container name.
3. Host workspace path, container workspace path, and working directory.
4. Required mounts, ports or host networking, devices, capabilities, and
   environment variables.
5. Whether GPU, GUI/X11, SSH forwarding, and persistent shell history are
   actually needed.
6. The interactive shell available in the image (`bash` or `sh`).

### Required behavior

- Every script MUST begin with `#!/usr/bin/env bash` and
  `set -euo pipefail`.
- Scripts that use repository paths MUST resolve them from
  `${BASH_SOURCE[0]}`, not from the caller's working directory.
- The build script MUST only build the image.
- The spawn script MUST refuse to overwrite an existing named container.
- The spawn script MUST run detached and use a long-lived, non-interactive
  primary command such as `sleep infinity`.
- The start script MUST distinguish missing, running, and stopped containers.
- The attach script MUST use `docker exec -it`; it MUST NOT use
  `docker attach`.
- Exiting a shell created by the attach script MUST leave the container running.
- Error messages SHOULD name the next appropriate lifecycle command.
- Creation-time configuration MUST remain in the spawn script. The start and
  attach scripts MUST NOT imply that they can alter that configuration.
- Scripts SHOULD quote all variable expansions and use a Bash array for Docker
  arguments.
- Optional features SHOULD be detected and should degrade with a clear message
  when unavailable.

### Adaptation procedure

1. Read the target Dockerfile and determine its shell and default command.
2. Choose image and container names; use the same constants in every script.
3. Implement the build script with the correct build context and Dockerfile.
4. Implement the spawn script with all creation-time options and a detached
   persistent command.
5. Implement the start script with explicit state checks.
6. Implement the attach script with an independent interactive exec session.
7. Make all scripts executable.
8. Validate syntax with `bash -n build.sh spawn.sh run.sh attach.sh`.
9. Test the full state sequence: missing, created/running, attached/exited,
   stopped, and restarted.

### Acceptance checklist

- `spawn.sh` leaves a running container after its own process exits.
- A second `spawn.sh` invocation fails without replacing the container.
- Exiting `attach.sh` does not stop the container.
- `run.sh` starts a stopped container and safely handles an already-running one.
- `attach.sh` reports a clear error for a missing or stopped container.
- Host-mounted project changes are visible inside the container.
- Required GPU and GUI behavior works, or the scripts clearly report why it is
  unavailable.
- Rebuilding the image and explicitly recreating the container uses the new
  image.

## Project-specific constants

Use this table when adapting the scripts rather than blindly copying values:

| Setting | LavaURMA value |
| --- | --- |
| Dockerfile | `docker/Dockerfile` |
| Build context | `one_policy_to_run_them_all` repository root |
| Image | `lavaurma` |
| Container | `lavaurma_dev` |
| Host workspace | Parent directory containing the related robotics repositories |
| Container workspace | `/app` |
| Working directory | `/app/one_policy_to_run_them_all` |
| Persistent command | `sleep infinity` |
| Interactive shell | `bash` |
| History mount | `dev/bash_history` to `/root/.bash_history` |
| Networking | Host network |
| GPU | Enabled when the NVIDIA Docker runtime is detected |
| GUI | Local or SSH-forwarded X11 when available |
