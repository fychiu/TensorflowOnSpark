#!/bin/bash

if [ $# != 3 ]; then
    echo "Please input CLUSTER_FILE USERNAME PUBLIC_KEY"
    exit 1
fi

CLUSTER_FILE=$1
USER=$2
PUBKEY=$3
SPARK_DIR="spark-2.2.0-bin-hadoop2.7"

NUM=0
MASTER_ADDR=""
GET_IP="/bin/hostname -I"
while read f
do
    if [ "$f" != "" ]; then
        MASTER_ADDR="$(cut -d' ' -f2 <<< $(ssh -i $PUBKEY $USER@${f} $GET_IP))"
        echo "Master address="
        echo $MASTER_ADDR
    fi
done < "$CLUSTER_FILE"


NUM=0
while read f
do
    if [ "$f" != "" ]; then
        if [ "$NUM" == 0 ]; then
            echo "$f master"

            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
                $SPARK_DIR/sbin/stop-master.sh
		exit
            " < /dev/null 
	else
	    echo "$f slave"

            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
                $SPARK_DIR/sbin/stop-slave.sh
		exit
            " < /dev/null 
	fi
    fi
    NUM=$((NUM+1))
done < "$CLUSTER_FILE"

echo "All the machines stop"
