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

/**
 * Blocking based deduplicator
 * @author pranab
 *
 */
object BlockingBasedDeduplicator {

  def main(args: Array[String]): Unit = {
	val Array(master: String, inputPath: String, outputPath: String, configFile: String) = args.length match {
		case x: Int if x == 4 => args.take(4)
		case _ => throw new IllegalArgumentException("missing command line args")
	}
	
	//load configuration
	System.setProperty("config.file", configFile)
	val config = ConfigFactory.load()
	
	val sparkConf = new SparkConf()
		.setMaster(master)
		.setAppName("StructuredTextAnalyzer")
		.set("spark.executor.memory", "1g")
	val sparkCntxt = new SparkContext(sparkConf)
		
	//add jars
	sparkCntxt.addJar(config.getString("sifarish.jar"))
	sparkCntxt.addJar(config.getString("jackson.core.jar"))
	sparkCntxt.addJar(config.getString("jackson.module.jar"))
	sparkCntxt.addJar(config.getString("lucene.core.jar"))
	sparkCntxt.addJar(config.getString("lucene.analyzers.common.jar"))
	sparkCntxt.addJar(config.getString("commons.lang.jar"))
		
	//config params
	val bucketFieldOrdinals = config.getIntList("bucket.field.ordinals")
	val concatenateBucketFields = config.getBoolean("concatenate.bucket.fields")
		
  }

}