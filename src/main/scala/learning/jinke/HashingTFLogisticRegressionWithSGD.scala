package learning.jinke

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.{DateTime, Duration}

/**
  *
  *垃圾分类: TF-IDF 特征
  * author: jinke
  * data: 2017/8/15
  **/
object HashingTFLogisticRegressionWithSGD {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf()
      .setAppName("dad").setMaster("local"))

    val spam = sc.textFile("data/spam.txt")
    val normal = sc.textFile("data/normal.txt")

    //dimension of feature is 10000
    val tf = new HashingTF(numFeatures = 10000)
    val idf = new IDF()

    //parameter: RDD sequence
    val spamFeature = spam.map(email => tf.transform(email.split(" ")))
    val idfModel = idf.fit(spamFeature)
    val tfIdfFeatureSpam = idfModel.transform(spamFeature).map(LabeledPoint(1, _))

    val normalFeature = normal.map(email => tf.transform(email.split(" ")))
    val idfModelN = idf.fit(normalFeature)
    val tfIdfFeatureNormal = idfModelN.transform(normalFeature).map(LabeledPoint(0, _))

    val trainData = tfIdfFeatureSpam.union(tfIdfFeatureNormal)
    trainData.cache()

    //use LogisticRegressionWithSGD to create model
    val start = new DateTime()
    val model = new LogisticRegressionWithSGD().run(trainData)
    val end = new DateTime()
    val duration = new Duration(start, end)
    println(s"=========== train time : ${duration.getStandardSeconds} seconds===============")

    // Test on a positive example (spam) and a negative one (normal).
    // First apply the same HashingTF feature transformation used on the training data.
    val posTestExample = tf.transform("O M G GET cheap stuff by sending money to ...".split(" "))
    val negTestExample = tf.transform("Hi Dad, I started studying Spark the other ...".split(" "))
    // Now use the learned model to predict spam/ham for new emails.
    println(s"Prediction for positive test example: ${model.predict(posTestExample)}")
    println(s"Prediction for negative test example: ${model.predict(negTestExample)}")
  }
}
