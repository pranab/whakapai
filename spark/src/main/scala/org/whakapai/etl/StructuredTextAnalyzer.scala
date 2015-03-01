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

package org.whakapai.etl

import java.io.StringReader
import scala.Array.canBuildFrom
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.lucene.util.Version
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.sifarish.etl.CountryStandardFormat
import org.sifarish.feature.SingleTypeSchema
import org.sifarish.util.Field
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.typesafe.config.ConfigFactory
import org.whakapai.common.JobConfiguration

/**
 * Normalizes various structured text field to bring everything into a canonical
 * format
 * @author pranab
 *
 */
object StructuredTextAnalyzer extends JobConfiguration {
  /**
 * @param args
 */
  def main(args: Array[String]) {
    val Array(master: String, inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args)
	
	//configuration and spark context
	val config = createConfig(configFile)
	val sparkConf = createSparkConf(master, "StructuredTextAnalyzer")
	val sparkCntxt = new SparkContext(sparkConf)
	
	//add jars
	addJars(sparkCntxt, config, "sifarish.jar", "jackson.core.jar", "jackson.module.jar", "lucene.core.jar",
	    "lucene.analyzers.common.jar", "commons.lang.jar")

	val fieldDelimRegex = config.getString("field.delim.regex")
	val country  = config.getString("text.country")
	val lang = config.getString("text.language")
	val countryFormat = CountryStandardFormat.createCountryStandardFormat(country)
	
	val analyzer = lang match {
	  case "en" => new EnglishAnalyzer(Version.LUCENE_44)
	  case _ =>  throw new IllegalArgumentException("unsupported language:" + lang)
	  
	}
	
	//data schema
    val filePath = config.getString("raw.schema.file.path")
    val schemaString = scala.io.Source.fromFile(filePath).mkString
    val schema = fromJson[SingleTypeSchema](schemaString)
	    
    //process
    val filedDelim = config.getString("field.delim.regex")
	val file = sparkCntxt.textFile(inputPath)
	val processed = file.map(l => {
	  val items = l.split(filedDelim)
	  var processedItems = List[String]()
	  for ((item, index) <- items.zipWithIndex) {
	    val field = schema.getEntity().getFieldByOrdinal(index)
	    
	    if (null != field && field.getDataType().equals(Field.DATA_TYPE_TEXT)) {
	      val format =  field.getTextDataSubTypeFormat()
	      val processedItem = field.getDataSubType() match {
	        case Field.TEXT_TYPE_PERSON_NAME => countryFormat.personNameFormat(item)
	        case Field.TEXT_TYPE_STREET_ADDRESS => {
	            val newItem = countryFormat.caseFormat(item, format)
	        	countryFormat.streetAddressFormat(newItem)
	        }
	        case Field.TEXT_TYPE_STREET_ADDRESS_ONE => {
	            val newItem = countryFormat.caseFormat(item, format)
	        	countryFormat.streetAddressOneFormat(newItem)
	        }
	        case Field.TEXT_TYPE_STREET_ADDRESS_TWO => {
	            val newItem = countryFormat.caseFormat(item, format)
	        	countryFormat.streetAddressTwoFormat(newItem)
	        }
	        case Field.TEXT_TYPE_CITY => countryFormat.caseFormat(item, format)
	        case Field.TEXT_TYPE_STATE => countryFormat.stateFormat(item)
	        case Field.TEXT_TYPE_ZIP => countryFormat.caseFormat(item, format)
	        case Field.TEXT_TYPE_COUNTRY => countryFormat.caseFormat(item, format)
	        case Field.TEXT_TYPE_EMAIL_ADDR => countryFormat.emailFormat(item, format)
	        case Field.TEXT_TYPE_PHONE_NUM => countryFormat.phoneNumFormat(item, format)
	        case _ => tokenize(item, analyzer)
	      }
	      processedItems = processedItem :: processedItems
	    }
	  }
	  processedItems = processedItems.reverse
	  processedItems.mkString(filedDelim)
	})
	processed.saveAsTextFile(outputPath)
	
  }
  
  private def fromJson[T](json: String)(implicit m : Manifest[T]): T = {
    val mapper = new ObjectMapper() with ScalaObjectMapper
    mapper.readValue[T](json)
  }
  
  private def tokenize(text : String, analyzer : Analyzer) : String = {
    val stream = analyzer.tokenStream("contents", new StringReader(text));
    val stBld = new StringBuilder();

    stream.reset();
    val termAttribute = stream.getAttribute(classOf[CharTermAttribute]).asInstanceOf[CharTermAttribute]
    while (stream.incrementToken()) {
		val token = termAttribute.toString();
		stBld.append(token).append(" ");
	} 
	stream.end();
	stream.close();
	stBld.toString();
  }
  
	
}