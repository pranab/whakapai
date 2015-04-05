#!/bin/bash

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi

PROJECT_HOME=/Users/pranab/Projects
JAR_NAME=$PROJECT_HOME/whakapai/spark/target/scala-2.10/whakapai-spark_2.10-1.0.jar
MASTER=spark://Pranab-Ghoshs-MacBook-Pro.local:7077
SIFARISH=file:///Users/pranab/Projects/sifarish/target/sifarish-1.0.jar
JACKSON_CORE=file:///Users/pranab/Projects/lib/jackson-core-asl-1.9.13.jar
JACKSON_MAPPER=file:///Users/pranab/Projects/lib/jackson-mapper-asl-1.9.13.jar
LUCENE_CORE=file:///Users/pranab/Projects/lib/lucene-core-4.4.0.jar
LUCENE_ANALYZERS_COMMON=file:///Users/pranab/Projects/lib/lucene-analyzers-common-4.4.0.jar
COMMONS_LANG=file:///Users/pranab/Projects/lib/commons-lang3-3.1.jar
COMMONS_MATH=file:///Users/pranab/Projects/lib/commons-math3-3.3.jar
LIB_JARS=$SIFARISH,$JACKSON_CORE,$JACKSON_MAPPER,$LUCENE_CORE,$LUCENE_ANALYZERS_COMMON,$COMMONS_LANG,$COMMONS_MATH

case "$1" in

"normalizeText") 
	CLASS_NAME=org.whakapai.etl.StructuredTextAnalyzer
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME  --jars $LIB_JARS --conf spark.ui.killEnabled=true $JAR_NAME $MASTER etl.properties
	;;
	
	