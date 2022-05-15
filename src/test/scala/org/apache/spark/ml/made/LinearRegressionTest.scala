package org.apache.spark.ml.made

import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.0001
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val points: DataFrame = LinearRegressionTest._points
  lazy val labels: Seq[Double] = LinearRegressionTest._labels
  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors

  "Model" should "predict" in {
//    System.setProperty("hadoop.home.dir", "D:\\Program Files\\winutils-master\\hadoop-3.0.0");

    val model = new LinearRegressionModel(Vectors.dense(1.0, 2.0), 1.0)
      .setFeaturesCol("features")

    val result: Array[Double] = model.transform(points).collect().map(_.getAs[Double](model.getPredictionCol))

    result(0) should be(labels(0))
    result(1) should be(labels(1))
    result(2) should be(labels(2))
    result(3) should be(labels(3))
  }

  "Estimator" should "calculate regression" in {
    val estimator = constructDefaultRegression()

    val model = estimator.fit(data)

    validateModelParams(model)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = constructDefaultRegression()

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(constructDefaultRegression()))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModelParams(model)
    validateModel(model, model.transform(data))
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(constructDefaultRegression()))
    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    val reReadModel = model.stages(0).asInstanceOf[LinearRegressionModel]
    validateModelParams(reReadModel)
    validateModel(reReadModel, reRead.transform(data))
  }

  private def validateModelParams(model: LinearRegressionModel) = {
    model.coefficients(0) should be(1.0 +- delta)
    model.coefficients(1) should be(2.0 +- delta)
    model.b should be(1.0 +- delta)
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val vectors: Array[Double] = data.collect().map(_.getAs[Double](2))

    vectors.length should be(4)

    vectors(0) should be(labels(0) +- delta)
    vectors(1) should be(labels(1) +- delta)
    vectors(2) should be(labels(2) +- delta)
    vectors(3) should be(labels(3) +- delta)
  }

  private def constructDefaultRegression() = {
    new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("labels")
      .setMaxIterCol(5000)
      .setTolCol(0.000000001)
      .setRegCol(0.0)
  }
}

object LinearRegressionTest extends WithSpark {
  lazy val _vectors = Seq(
    Vectors.dense(13.5, 12, 38.5),
    Vectors.dense(-1.0, 0, 0),
    Vectors.dense(20.0, 5, 31),
    Vectors.dense(-10.0, 2, -5)
  )

  lazy val _points: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(Vectors.dense(x(0), x(1))))
      .toDF("features")
  }

  lazy val _labels: Seq[Double] = {
    _vectors.map(x => x(2))
  }

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple2(Vectors.dense(x(0), x(1)), x(2)))
      .toDF("features", "labels")
  }
}
