#!/bin/bash

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi

PROJECT_HOME=/Users/pranab/Projects
JAR_NAME=$PROJECT_HOME/whakapai/spark/target/scala-2.10/whakapai-spark_2.10-1.0.jar
MASTER=spark://Pranab-Ghoshs-MacBook-Pro.local:7077

case "$1" in

"normalizeText") 
	CLASS_NAME=org.whakapai.etl.StructuredTextAnalyzer
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   --conf spark.ui.killEnabled=true $JAR_NAME $MASTER etl.properties
	;;
	
	