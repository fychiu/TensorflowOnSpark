#!/bin/bash
CLUSTER_FILE=$1
USER=$2
PUBKEY=$3
SPARK_URL="http://apache.cs.utah.edu/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz"

#exec 1>/dev/null

NUM=0
while read f
do
    if [ "$f" != "" ]; then
        if [ "$NUM" == 0 ]; then
            echo "$f master"
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
	fi
	echo ""
	NUM=$((NUM+1))
    fi
done < "$CLUSTER_FILE"


while read f
do
    if [ "$f" != "" ]; then
        echo "$f master"
        ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
            sudo apt-get update
            sudo apt-get install oracle-java8-installer
            sudo apt-get install openjdk-8-jdk openjdk-8-jre
            echo "export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-arm64"
            source ~/.bashrc
            java -version
            /bin/java -version
            
            wget $SPARK_URL
            tar -xvf spark-2.2.0-bin-hadoop2.7.tgz
            
            echo "export SPARK_HOME=/users/Fangyi/spark-2.2.0-bin-hadoop2.7" >> ~/.bashrc
            echo "export PATH=${SPARK_HOME}/bin:$PATH" >> ~/.bashrc
            source ~/.bashrc
            
            sudo apt-get install python-software-properties
            sudo apt-get install scala
            " < /dev/null 
    fi
done < "$CLUSTER_FILE"
