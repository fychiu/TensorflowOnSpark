#!/bin/bash
sudo apt-get update
sudo apt-get install oracle-java8-installer
sudo apt-get install openjdk-8-jdk openjdk-8-jre
echo "export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-arm64"
source ~/.bashrc
java -version
/bin/java -version

wget http://apache.cs.utah.edu/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz
tar -xvf spark-2.2.0-bin-hadoop2.7.tgz

echo "export SPARK_HOME=/users/Fangyi/spark-2.2.0-bin-hadoop2.7" >> ~/.bashrc
echo "export PATH=${SPARK_HOME}/bin:$PATH" >> ~/.bashrc
source ~/.bashrc

sudo apt-get install python-software-properties
sudo apt-get install scala
