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

package org.ruscello.similarity

import scala.collection.JavaConversions._
import org.apache.spark.SparkConf
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import com.typesafe.config.ConfigFactory
import org.apache.spark.streaming.dstream.PairDStreamFunctions
import org.hoidla.window.SizeBoundWindow
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.SparkContext
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

object StructuredTextAnalyzer {
  /**
 * @param args
 */
  def main(args: Array[String]) {
	val Array(master: String, inputPath: String, configFile: String) = args.length match {
		case x: Int if x == 3 => args.take(2)
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
	val fieldDelimRegex = config.getString("field.delim.regex")
	val country  = config.getString("text.country")
	val lang = config.getString("text.language")
	
	val analyzer = lang match {
	  case "en" => new EnglishAnalyzer(Version.LUCENE_44)
	  case _ =>  throw new IllegalArgumentException("unsupported language:" + lang)
	  
	}
	    
	val file = sparkCntxt.textFile(inputPath)
	file.map(l => {
	  
	})
	
  }
	
}