package learning.jinke

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.linalg.Vectors

object test {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf()
      .setAppName("dad22").setMaster("local"))

    val denseVector = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
    val denseVector2 = Vectors.dense(2.0, 3.0, 4.0, 5.0, 6.0)
    //    val sparseVector = Vectors.sparse(10, Array(0,2,6),Array(1.0, 2.0, 3.0))

    println(denseVector.toArray.mkString("  "))
    val RDD1 = sc.parallelize(List(denseVector,denseVector2))
    val scaler = new StandardScaler(withMean = true, withStd = true)
    val model = scaler.fit(RDD1)
    val result = model.transform(RDD1)

    val normalized = new Normalizer().transform(result)
    normalized.foreach(fea=>println(fea.toArray.mkString("  ")))

    //    println(sparseVector.toArray.mkString("  "))
  }
}
