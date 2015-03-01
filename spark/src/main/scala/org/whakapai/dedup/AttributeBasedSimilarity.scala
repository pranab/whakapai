/*
 * whakapai: etl on spark
 * Author: Pranab Ghosh
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You may
 * obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package org.whakapai.dedup

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import com.typesafe.config.ConfigFactory

import org.whakapai.common.JobConfiguration

/**
 * Blocking based deduplicator
 * @author pranab
 *
 */
object AttributeBasedSimilarity extends JobConfiguration {

  def main(args: Array[String]): Unit = {
    val Array(master: String, inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args)
	
	//configuration and spark context
	val config = createConfig(configFile)
	val sparkConf = createSparkConf(master, "Blocking based similarity")
	val sparkCntxt = new SparkContext(sparkConf)
	
	//add jars
	addJars(sparkCntxt, config, "sifarish.jar", "jackson.core.jar", "jackson.module.jar", "lucene.core.jar",
	    "lucene.analyzers.common.jar", "commons.lang.jar")
		
	//config params
	val bucketFieldOrdinals = config.getIntList("bucket.field.ordinals")
	val concatenateBucketFields = config.getBoolean("concatenate.bucket.fields")
	val recordWiseSimilarity = config.getBoolean("record.wise.similarity")
		
  }

}