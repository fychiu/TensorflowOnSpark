#!/bin/bash

if [$# != 3]; then
    echo "Please input CLUSTER_FILE USERNAME PUBKEY"
fi

CLUSTER_FILE=$1
USER=$2
PUBKEY=$3
SPARK_URL="http://apache.cs.utah.edu/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz"
MASTER_ADDR=""

#exec 1>/dev/null

HOST_ADDR=""

NUM=0
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
            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${HOST_ADDR}" "
	        ssh -T -o \"StrictHostKeyChecking no\" ${USER}@${f}
		exit
		"  < /dev/null
	fi
	echo ""
	NUM=$((NUM+1))
    fi
done < "$CLUSTER_FILE"


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
            
            echo \"export SPARK_HOME=/users/Fangyi/spark-2.2.0-bin-hadoop2.7\" >> ~/.bashrc
            exec bash
            echo \"export PATH=\$SPARK_HOME/bin:\$PATH\" >> ~/.bashrc
            exec bash
            
            " < /dev/null 
    fi
    NUM=$((NUM+1))
done < "$CLUSTER_FILE"


NUM=0
while read f
do
    if [ "$f" != "" ]; then
        if [ "$NUM" == 0 ]; then
            echo "$f master"
            MASTER_ADDR=$(ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "curl ifconfig.me")

            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
                
                cp \$SPARK_HOME/conf/spark-env.sh-template \$SPARK_HOME/conf/spark-env.sh
                echo SPARK_MASTER_HOST=${MASTER_ADDR} >> \$SPARK_HOME/conf/spark-env.sh
                \$SPARK_HOME/sbin/start-master.sh

                echo "Master address="
                echo $MACHINE_ADDR
            " < /dev/null 
	else
	    echo "$f slave"

            ssh -o "StrictHostKeyChecking no" -i $PUBKEY "${USER}@${f}" "
                cp \$SPARK_HOME/conf/spark-env.sh-template \$SPARK_HOME/conf/spark-env.sh
                echo SPARK_MASTER_HOST=${MASTER_ADDR} >> \$SPARK_HOME/conf/spark-env.sh
                \$SPARK_HOME/sbin/start-slave.sh
            " < /dev/null 
	fi

    fi
    NUM=$((NUM+1))
done < "$CLUSTER_FILE"

echo "Please use browser to connect to \n\n\t\t http://${MASTER_ADDR}:8081"
