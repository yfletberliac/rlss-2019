#!/usr/bin/env bash
IMAGE_NAME=shmuma/miniwob

count=`docker ps -q | wc -l`

if test $count -ne 0 ; then
    echo You already have $count containers running, are you sure you want more?
    exit
fi

for i in `seq 1 ${1:-1}`; do
    echo Starting container $i
    P1=$((5900+$i-1))
    P2=$((15900+$i-1))
    docker run -d -p $P1:5900 -p $P2:15900 --privileged --ipc host --cap-add SYS_ADMIN $IMAGE_NAME run -f 5
done
