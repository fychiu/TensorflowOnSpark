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
