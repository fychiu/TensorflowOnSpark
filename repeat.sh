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
	else
	    echo "$f slave"
            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
	        sudo apt-get -y install libffi-dev
		sudo pip install cffi
		sudo pip install soundfile
		sudo pip install python_speech_features
		sudo pip install Whoosh
                exit
                "  < /dev/null
	fi
	echo ""
	NUM=$((NUM+1))
    fi
done < "$CLUSTER_FILE"

