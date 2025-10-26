#!/bin/bash
set -e

usage()
{
    echo "Usage: ./start_container.sh [OPTION...]"
    echo ""
    echo "  -h, --help        Give this help list"
    echo "    , --image       Select the image name of the container"
    echo "  -v, --volume      Bind mount a volume into the container"
    echo "    , --name        Assign a name to the container"
	echo "  -d, --detach      Run container in background and print container ID"
    exit 1
}

image_name=kds285/minizero:latest
container_volume=""
container_argumenets=""
use_default_volume=true
while :; do
	case $1 in
		-h|--help) shift; usage
		;;
		--image) shift; image_name=${1}
		;;
		-v|--volume) shift; container_volume="${container_volume} -v ${1}"; use_default_volume=false
		;;
        --name) shift; container_argumenets="${container_argumenets} --name ${1}"
		;;
		-d|--detach) container_argumenets="${container_argumenets} -d"
		;;
		"") break
		;;
		*) echo "Unknown argument: $1"; usage
		;;
	esac
	shift
done

# Add default volume if no custom volume specified
if [ "$use_default_volume" = true ]; then
    container_volume="-v .:/workspace"
fi

container_argumenets=$(echo ${container_argumenets} | xargs)

# Use docker if podman is not available
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "Error: Neither podman nor docker is installed"
    exit 1
fi

echo "${CONTAINER_CMD} run ${container_argumenets} --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host --ipc=host --rm -it ${container_volume} ${image_name}"
${CONTAINER_CMD} run ${container_argumenets} --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host --ipc=host --rm -it ${container_volume} ${image_name}
