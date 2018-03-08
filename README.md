# TensorflowOnSpark

To use `setup-ssh.sh` to setup the whole cluster, three things are required.
1. A `.txt` file to indicate the machines address with the master address at first line and others are the following line.
2. A USERNAME for all the machines. Currently, we only support cluster machines with identical usernames.
3. A RSA public key file to let your local machine access all the cluster machines. Please refer other resources to know how to generate a public key.

Please use the file as the following way:
```
$> ./setup-all.sh CLUSTER_MACHINES_FILE.txt USERNAME RSA_PUBLIC_KEY_FILE  
```

After a while, you can see the WebUI address for the Spark cluster. Enjoy it!  

Note:  
There are three additional files functioning on different parts of setup:  

`setup-ssh.sh: setup ssh connection between master and slaves machines.`  
`setup-install.sh: install Spark on every machine.`  
`setup-start.sh: launch Spark on master and connect slaves to the master.`  

There might be a `stop.sh` to trun down every machines. To be continued.
