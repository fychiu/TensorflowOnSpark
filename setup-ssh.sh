#!/bin/bash

if [ $# != 3 ]; then
    echo "Please input CLUSTER_FILE USERNAME PUBLIC_KEY"
    exit 1
fi

CLUSTER_FILE=$1
USER=$2
PUBKEY=$3
SPARK_URL="http://apache.cs.utah.edu/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz"
SPARK_DIR="spark-2.2.0-bin-hadoop2.7"

#exec 1>/dev/null

NUM=0
HOST_ADDR=""
while read f
do
    if [ "$f" != "" ]; then
        if [ "$NUM" == 0 ]; then
            echo "$f master"
	    HOST_ADDR=$f
            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
	        rm -f ~/.ssh/id_rsa
                cd ~/.ssh
                ssh-keygen -f id_rsa -t rsa -N ''
                cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
                " < /dev/null
	    scp -i $PUBKEY ${USER}@${f}:~/.ssh/id_rsa.pub ./
	else
	    echo "$f slave"
	    scp -i $PUBKEY ./id_rsa.pub ${USER}@${f}:~/.ssh/
            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
                cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
                exit
                " < /dev/null
            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@$HOST_ADDR" "
	        ssh -T -o \"StrictHostKeyChecking no\" ${USER}@${f}
		exit
		"  < /dev/null
	fi
	echo ""
	NUM=$((NUM+1))
    fi
done < "$CLUSTER_FILE"
