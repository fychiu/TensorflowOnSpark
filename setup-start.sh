#!/bin/bash

if [ $# != 3 ]; then
    echo "Please input CLUSTER_FILE USERNAME PUBLIC_KEY"
    exit 1
fi

CLUSTER_FILE=$1
USER=$2
PUBKEY=$3

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
        ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
	    cd $SPARK_DIR/conf
            cp spark-env.sh.template spark-env.sh
            echo spark_master_host=${MASTER_ADDR} >> spark-env.sh
        " < /dev/null 

        if [ "$NUM" == 0 ]; then
            echo "$f master"

            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
                $SPARK_DIR/sbin/start-master.sh
		exit
            " < /dev/null 
	else
	    echo "$f slave"

            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
                $SPARK_DIR/sbin/start-slave.sh spark://$MASTER_ADDR:7077
		exit
            " < /dev/null 
	fi
    fi
    NUM=$((NUM+1))
done < "$CLUSTER_FILE"

echo "Please use browser to connect to

                                           http://${MASTER_ADDR}:8081"
