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

NUM=0
while read f
do
    if [ "$f" != "" ]; then
        if [ "$NUM" == 0 ]; then
            echo "$f master"
	else
	    echo "$f slave"
	fi
        ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
            sudo apt-get -y update
            sudo apt-get -y install openjdk-8-jdk openjdk-8-jre
            sudo apt-get -y install scala python-software-properties
            echo \"export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-arm64\"
            source ~/.bashrc
            java -version
            
            wget $SPARK_URL
            tar -xvf spark-2.2.0-bin-hadoop2.7.tgz
            
            echo \"export SPARK_HOME=\\\$HOME/${SPARK_DIR}\" >> ~/.bashrc
            echo export PATH=\\\$SPARK_HOME/bin:\\\$PATH >> ~/.bashrc
            exit 
            " < /dev/null 
    fi
    NUM=$((NUM+1))
done < "$CLUSTER_FILE"
